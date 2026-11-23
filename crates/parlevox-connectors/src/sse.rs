//! Byte-oriented Server-Sent Events parsing.
//!
//! The parser is fed arbitrary byte chunks because the network does not
//! respect event boundaries. It buffers until a complete line exists, decodes
//! that line as UTF-8, and dispatches an [`SseEvent`] on a blank line. A
//! chunk-split anywhere produces the same event sequence as one-shot input.
//!
//! The buffer is bounded: an unterminated line larger than the configured
//! bound is [`ConnectorError::Capacity`], never an ever-growing allocation.

use crate::error::ConnectorError;

/// One dispatched SSE event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// The `event:` field, when the provider sent one.
    pub event: Option<String>,
    /// The `data:` fields joined with newlines.
    pub data: String,
    /// The last `id:` field, when present.
    pub id: Option<String>,
}

/// A streaming SSE decoder.
#[derive(Debug)]
pub struct SseParser {
    buffer: Vec<u8>,
    max_buffer_bytes: usize,
    event: Option<String>,
    data: Vec<String>,
    id: Option<String>,
}

impl SseParser {
    /// Create a parser whose unterminated line may not exceed
    /// `max_buffer_bytes`. A zero bound rejects any unterminated data.
    #[must_use]
    pub fn new(max_buffer_bytes: usize) -> Self {
        SseParser {
            buffer: Vec::with_capacity(max_buffer_bytes.min(4096)),
            max_buffer_bytes,
            event: None,
            data: Vec::new(),
            id: None,
        }
    }

    /// Feed one arbitrary byte chunk and return every event it completed.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectorError::Protocol`] when a field is not UTF-8 and
    /// [`ConnectorError::Capacity`] when a line exceeds the configured bound.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, ConnectorError> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        let mut consumed = 0usize;
        while let Some(relative) = self.buffer[consumed..]
            .iter()
            .position(|&byte| byte == b'\n')
        {
            let end = consumed.saturating_add(relative);
            self.check_line_bound(end.saturating_sub(consumed))?;
            let mut line = self.buffer[consumed..end].to_vec();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            self.process_line(&line, &mut events)?;
            consumed = end.saturating_add(1);
        }
        self.buffer.drain(..consumed);
        if self.buffer.len() > self.max_buffer_bytes {
            return Err(ConnectorError::Capacity(format!(
                "SSE line exceeds {} bytes",
                self.max_buffer_bytes
            )));
        }
        Ok(events)
    }

    /// Flush the unterminated tail and dispatch any pending event.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectorError::Protocol`] when the tail is not UTF-8.
    pub fn finish(&mut self) -> Result<Vec<SseEvent>, ConnectorError> {
        let mut events = Vec::new();
        if !self.buffer.is_empty() {
            let mut line = std::mem::take(&mut self.buffer);
            self.check_line_bound(line.len())?;
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            self.process_line(&line, &mut events)?;
        }
        self.dispatch(&mut events);
        Ok(events)
    }

    fn check_line_bound(&self, length: usize) -> Result<(), ConnectorError> {
        if length > self.max_buffer_bytes {
            return Err(ConnectorError::Capacity(format!(
                "SSE line exceeds {} bytes",
                self.max_buffer_bytes
            )));
        }
        Ok(())
    }

    fn process_line(
        &mut self,
        line: &[u8],
        events: &mut Vec<SseEvent>,
    ) -> Result<(), ConnectorError> {
        if line.is_empty() {
            self.dispatch(events);
            return Ok(());
        }
        if line.first() == Some(&b':') {
            return Ok(());
        }
        let (field, value) = match line.iter().position(|&byte| byte == b':') {
            Some(index) => (
                &line[..index],
                strip_one_space(&line[index.saturating_add(1)..]),
            ),
            None => (line, &line[line.len()..]),
        };
        let field = std::str::from_utf8(field).map_err(invalid_utf8)?;
        let value = std::str::from_utf8(value).map_err(invalid_utf8)?;
        match field {
            "data" => self.data.push(value.to_owned()),
            "event" => self.event = Some(value.to_owned()),
            "id" => self.id = Some(value.to_owned()),
            // `retry` is scheduling advice we deliberately ignore: retry
            // pacing belongs to RetryPolicy, not to the provider.
            _ => {}
        }
        Ok(())
    }

    fn dispatch(&mut self, events: &mut Vec<SseEvent>) {
        let event = self.event.take();
        let id = self.id.take();
        let data = if self.data.is_empty() {
            String::new()
        } else {
            self.data.join("\n")
        };
        self.data.clear();
        if data.is_empty() && event.is_none() {
            return;
        }
        events.push(SseEvent { event, data, id });
    }
}

fn strip_one_space(value: &[u8]) -> &[u8] {
    match value.first() {
        Some(b' ') => &value[1..],
        _ => value,
    }
}

fn invalid_utf8(_error: std::str::Utf8Error) -> ConnectorError {
    ConnectorError::Protocol("SSE field is not valid UTF-8".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_all(input: &[u8]) -> Vec<SseEvent> {
        let mut parser = SseParser::new(4096);
        let mut events = parser.push(input).expect("valid input");
        events.extend(parser.finish().expect("valid tail"));
        events
    }

    #[test]
    fn multi_line_data_and_event_names_are_joined() {
        let events = parse_all(b"event: delta\ndata: one\ndata: two\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.as_deref(), Some("delta"));
        assert_eq!(events[0].data, "one\ntwo");
    }

    #[test]
    fn comments_and_crlf_are_handled() {
        let events = parse_all(b": keep-alive\r\ndata: hi\r\n\r\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hi");
    }

    #[test]
    fn chunk_splits_do_not_change_the_event_sequence() {
        let input = b"data: a\n\ndata: b\n\n";
        let whole = parse_all(input);
        for split in 0..=input.len() {
            let mut parser = SseParser::new(4096);
            let mut events = parser.push(&input[..split]).expect("head");
            events.extend(parser.push(&input[split..]).expect("tail"));
            events.extend(parser.finish().expect("finish"));
            assert_eq!(events, whole, "split at {split}");
        }
    }

    #[test]
    fn invalid_utf8_is_protocol_not_panic() {
        let mut parser = SseParser::new(4096);
        let error = parser.push(b"data: \xff\n\n").expect_err("invalid UTF-8");
        assert!(matches!(error, ConnectorError::Protocol(_)));
    }

    #[test]
    fn overlong_line_is_capacity() {
        let mut parser = SseParser::new(4);
        let error = parser
            .push(b"data: long line without newline")
            .expect_err("bounded");
        assert!(matches!(error, ConnectorError::Capacity(_)));
    }
}
