//! Credential values and auth injection.
//!
//! [`Secret`] is the only place a credential lives. It has no `Display`, no
//! `Serialize` and a redacting `Debug`, so the usual ways a value leaks into a
//! log or a fixture are closed by construction. [`Auth::apply`] is the one
//! place a secret is turned into a wire form: an `Authorization` header, a
//! custom header, or a templated query parameter.

use std::fmt;

use crate::error::ConnectorError;
use crate::transport::{is_valid_header_name, is_valid_header_value, RequestSpec};

/// The placeholder a query template substitutes with the credential.
pub const QUERY_TEMPLATE_PLACEHOLDER: &str = "{key}";

/// A credential value.
///
/// Deliberately opaque: `Debug` prints `Secret(<redacted>)`, and there is no
/// `Display`, `Serialize` or `Deref`, so formatting a secret by accident is a
/// compile error rather than a leak.
#[derive(Clone)]
pub struct Secret(String);

impl Secret {
    /// Wrap a credential value.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Secret(value.into())
    }

    /// Whether the value is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The raw value. Crate-internal: only auth injection and the factory use
    /// it, and neither puts it in a message.
    pub(crate) fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(<redacted>)")
    }
}

/// How a connector authenticates.
///
/// Non-exhaustive: construct values with the associated functions so the
/// validation rules cannot be bypassed.
#[derive(Clone)]
#[non_exhaustive]
pub enum Auth {
    /// No auth injection.
    None,
    /// `Authorization: Bearer <token>`.
    Bearer {
        /// The credential.
        token: Secret,
    },
    /// A bearer-equivalent credential in a provider-specific header.
    Header {
        /// Header name.
        name: String,
        /// Header value.
        value: Secret,
    },
    /// A credential substituted into a query parameter template.
    QueryTemplate {
        /// Query parameter name.
        param: String,
        /// Template containing [`QUERY_TEMPLATE_PLACEHOLDER`].
        template: String,
        /// The credential rendered into the template.
        token: Secret,
    },
}

impl Auth {
    /// No auth injection.
    #[must_use]
    pub fn none() -> Self {
        Auth::None
    }

    /// Bearer-token auth, the OpenAI family's default.
    #[must_use]
    pub fn bearer(token: Secret) -> Self {
        Auth::Bearer { token }
    }

    /// A custom header, e.g. `x-api-key`.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectorError::Protocol`] when `name` is not an HTTP token or
    /// the credential contains a character that cannot appear in a header
    /// value.
    pub fn custom_header(name: impl Into<String>, token: Secret) -> Result<Self, ConnectorError> {
        let name = name.into();
        if !is_valid_header_name(&name) {
            return Err(ConnectorError::Protocol(
                "invalid auth header name".to_owned(),
            ));
        }
        if !is_valid_header_value(token.expose()) {
            return Err(ConnectorError::Protocol(
                "credential is not a valid header value".to_owned(),
            ));
        }
        Ok(Auth::Header { name, value: token })
    }

    /// A query-parameter auth template such as `key={key}`.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectorError::Protocol`] when `param` is empty or
    /// `template` does not contain [`QUERY_TEMPLATE_PLACEHOLDER`].
    pub fn query_template(
        param: impl Into<String>,
        template: impl Into<String>,
        token: Secret,
    ) -> Result<Self, ConnectorError> {
        let param = param.into();
        let template = template.into();
        if param.is_empty() {
            return Err(ConnectorError::Protocol(
                "query template parameter name is empty".to_owned(),
            ));
        }
        if !template.contains(QUERY_TEMPLATE_PLACEHOLDER) {
            return Err(ConnectorError::Protocol(format!(
                "query template must contain {QUERY_TEMPLATE_PLACEHOLDER}"
            )));
        }
        Ok(Auth::QueryTemplate {
            param,
            template,
            token,
        })
    }

    /// Inject this auth into `request`.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectorError::Protocol`] when the resulting header name or
    /// value is not valid HTTP.
    pub fn apply(&self, request: &mut RequestSpec) -> Result<(), ConnectorError> {
        match self {
            Auth::None => Ok(()),
            Auth::Bearer { token } => {
                let value = format!("Bearer {}", token.expose());
                request.push_header("authorization", &value)
            }
            Auth::Header { name, value } => request.push_header(name, value.expose()),
            Auth::QueryTemplate {
                param,
                template,
                token,
            } => {
                let rendered = template.replace(QUERY_TEMPLATE_PLACEHOLDER, token.expose());
                request.query.push((param.clone(), rendered));
                Ok(())
            }
        }
    }
}

impl fmt::Debug for Auth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Auth::None => f.write_str("Auth::None"),
            Auth::Bearer { .. } => f.write_str("Auth::Bearer { token: <redacted> }"),
            Auth::Header { name, .. } => f
                .debug_struct("Auth::Header")
                .field("name", name)
                .field("value", &"<redacted>")
                .finish(),
            Auth::QueryTemplate {
                param, template, ..
            } => f
                .debug_struct("Auth::QueryTemplate")
                .field("param", param)
                .field("template", template)
                .field("token", &"<redacted>")
                .finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_is_injected_as_an_authorization_header() {
        let mut request = RequestSpec::post("https://example.invalid/v1");
        Auth::bearer(Secret::new("sk-test-abc"))
            .apply(&mut request)
            .expect("apply");
        assert_eq!(request.headers.len(), 1);
        assert_eq!(request.headers[0].name(), "authorization");
        assert_eq!(request.headers[0].value(), "Bearer sk-test-abc");
    }

    #[test]
    fn custom_header_and_query_template_are_injected() {
        let mut request = RequestSpec::post("https://example.invalid/v1");
        Auth::custom_header("x-api-key", Secret::new("sk-test-header"))
            .expect("valid")
            .apply(&mut request)
            .expect("apply");
        Auth::query_template("api_key", "prefix-{key}", Secret::new("sk-test-query"))
            .expect("valid")
            .apply(&mut request)
            .expect("apply");
        assert_eq!(request.headers[0].name(), "x-api-key");
        assert_eq!(request.headers[0].value(), "sk-test-header");
        assert_eq!(
            request.query,
            vec![("api_key".to_owned(), "prefix-sk-test-query".to_owned())]
        );
    }

    #[test]
    fn auth_debug_and_request_debug_redact_the_secret() {
        let secret = "sk-test-do-not-print-this";
        let auth = Auth::query_template("key", "{key}", Secret::new(secret)).expect("valid");
        let mut request = RequestSpec::post("https://example.invalid/v1");
        auth.apply(&mut request).expect("apply");
        assert!(!format!("{auth:?}").contains(secret));
        assert!(!format!("{request:?}").contains(secret));
    }

    #[test]
    fn malformed_auth_is_refused() {
        assert!(Auth::custom_header("bad name", Secret::new("x")).is_err());
        assert!(Auth::custom_header("x-key", Secret::new("line\nbreak")).is_err());
        assert!(Auth::query_template("", "{key}", Secret::new("x")).is_err());
        assert!(Auth::query_template("key", "no-placeholder", Secret::new("x")).is_err());
    }

    #[test]
    fn secret_debug_is_redacted() {
        assert_eq!(
            format!("{:?}", Secret::new("sk-test-abc")),
            "Secret(<redacted>)"
        );
    }
}
