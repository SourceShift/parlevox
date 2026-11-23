//! Newline-delimited JSON parsing.
//!
//! NDJSON bodies arrive in arbitrary chunks; this parser buffers until a
//! complete line exists and parses that line as one JSON value. Blank lines are
//! skipped. The buffered unterminated line is bounded, so malformed input
//! cannot grow memory without limit.

use serde_json::Value;

use crate::error::ConnectorError;

/// A streaming NDJSON decoder.
#[derive(Debug)]
pub struct NdjsonParser {
    buffer: Vec<u8>,
    max_buffer_bytes: usize,
}

impl NdjsonParser {
    /// Create a parser whose unterminated line may not exceed
    /// `max_buffer_bytes`.
    #[must_use]
    pub fn new(max_buffer_bytes: usize) -> Self {
        NdjsonParser {
            buffer: Vec::with_capacity(max_buffer_bytes.min(4096)),
            max_buffer_bytes,
        }
    }

    /// Feed one byte chunk and return every complete record it produced.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectorError::Protocol`] for invalid UTF-8 or invalid JSON,
    /// and [`ConnectorError::Capacity`] when a line exceeds the bound.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<Value>, ConnectorError> {
        self.buffer.extend_from_slice(chunk);
        let mut records = Vec::new();
        let mut consumed = 0usize;
        while let Some(relative) = self.buffer[consumed..]
            .iter()
            .position(|&byte| byte == b'\n')
        {
            let end = consumed.saturating_add(relative);
            self.check_line_bound(end.saturating_sub(consumed))?;
            let line = self.buffer[consumed..end].to_vec();
            parse_line(&line, &mut records)?;
            consumed = end.saturating_add(1);
        }
        self.buffer.drain(..consumed);
        if self.buffer.len() > self.max_buffer_bytes {
            return Err(ConnectorError::Capacity(format!(
                "NDJSON line exceeds {} bytes",
                self.max_buffer_bytes
            )));
        }
        Ok(records)
    }

    /// Parse an unterminated final line, if any.
    ///
    /// # Errors
    ///
    /// Same conditions as [`NdjsonParser::push`].
    pub fn finish(&mut self) -> Result<Vec<Value>, ConnectorError> {
        let mut records = Vec::new();
        if !self.buffer.is_empty() {
            let line = std::mem::take(&mut self.buffer);
            self.check_line_bound(line.len())?;
            parse_line(&line, &mut records)?;
        }
        Ok(records)
    }

    fn check_line_bound(&self, length: usize) -> Result<(), ConnectorError> {
        if length > self.max_buffer_bytes {
            return Err(ConnectorError::Capacity(format!(
                "NDJSON line exceeds {} bytes",
                self.max_buffer_bytes
            )));
        }
        Ok(())
    }
}

fn parse_line(line: &[u8], records: &mut Vec<Value>) -> Result<(), ConnectorError> {
    let text = std::str::from_utf8(line)
        .map_err(|_| ConnectorError::Protocol("NDJSON line is not valid UTF-8".to_owned()))?;
    if text.trim().is_empty() {
        return Ok(());
    }
    let value: Value = serde_json::from_str(text)
        .map_err(|error| ConnectorError::Protocol(format!("invalid NDJSON record: {error}")))?;
    records.push(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_split_across_chunks_parse() {
        let mut parser = NdjsonParser::new(4096);
        let mut records = parser.push(b"{\"a\":1}\n{\"b\"").expect("head");
        records.extend(parser.push(b":2}\n").expect("tail"));
        records.extend(parser.finish().expect("finish"));
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["a"], 1);
        assert_eq!(records[1]["b"], 2);
    }

    #[test]
    fn blank_lines_are_skipped() {
        let mut parser = NdjsonParser::new(4096);
        let records = parser.push(b"\n  \n{\"a\":1}\n\n").expect("valid");
        assert_eq!(records.len(), 1);
    }

    #[test]
    fn malformed_json_is_protocol() {
        let mut parser = NdjsonParser::new(4096);
        let error = parser.push(b"{not json}\n").expect_err("invalid");
        assert!(matches!(error, ConnectorError::Protocol(_)));
    }

    #[test]
    fn invalid_utf8_is_protocol() {
        let mut parser = NdjsonParser::new(4096);
        let error = parser.push(b"\xff\n").expect_err("invalid UTF-8");
        assert!(matches!(error, ConnectorError::Protocol(_)));
    }
}
