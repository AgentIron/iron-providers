//! Shared SSE (Server-Sent Events) parser
//!
//! Provides an event-aware SSE parser that correctly handles:
//! - Chunk boundaries splitting mid-line
//! - Multi-line `data:` fields joined into a single payload
//! - Comment lines (`:`)
//! - Event name lines (`event:`)
//! - Empty keepalive frames (blank line boundaries)
//!
//! Both Chat Completions and Anthropic adapters use this framing logic.

/// A single parsed SSE event containing the joined data payload and optional event name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// Optional event name from `event:` lines.
    pub event: Option<String>,
    /// Joined data payload from one or more `data:` lines.
    pub data: String,
}

/// Incremental SSE parser that buffers partial input and yields complete events.
///
/// SSE protocol rules (simplified):
/// - Events are separated by blank lines.
/// - Within an event, `data:` lines are concatenated with newlines.
/// - `event:` sets the event type.
/// - Lines starting with `:` are comments and ignored.
/// - `id:` and `retry:` lines are ignored by this parser (not needed).
#[derive(Debug, Default)]
pub struct SseParser {
    buffer: String,
    current_data_lines: Vec<String>,
    current_event: Option<String>,
}

impl SseParser {
    /// Create a new SSE parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed raw bytes into the parser and return any complete events.
    ///
    /// The bytes may contain partial lines, complete lines, or multiple events.
    /// The parser buffers internally and only returns events when a complete
    /// event boundary (blank line) is reached.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        let mut events = Vec::new();

        self.buffer.push_str(&String::from_utf8_lossy(bytes));

        while let Some(line_end) = self.buffer.find('\n') {
            let line = self.buffer[..line_end]
                .trim_end_matches('\r')
                .trim()
                .to_string();
            self.buffer = self.buffer[line_end + 1..].to_string();

            if line.is_empty() {
                // Blank line = event boundary
                if !self.current_data_lines.is_empty() {
                    events.push(SseEvent {
                        event: self.current_event.take(),
                        data: self.current_data_lines.join("\n"),
                    });
                    self.current_data_lines.clear();
                }
                // If no data lines, this was just a keepalive — skip
                continue;
            }

            if line.starts_with(':') {
                // Comment line — ignore
                continue;
            }

            if let Some(data) = line.strip_prefix("data:") {
                // `data:` with optional leading space
                let value = data.strip_prefix(' ').unwrap_or(data);
                self.current_data_lines.push(value.to_string());
            } else if let Some(event) = line.strip_prefix("event:") {
                let value = event.strip_prefix(' ').unwrap_or(event);
                self.current_event = Some(value.to_string());
            }
            // Ignore `id:` and `retry:` lines
        }

        events
    }

    /// Flush any remaining buffered data as a final event.
    ///
    /// Call this when the stream ends to emit any event that wasn't
    /// terminated by a blank line.
    pub fn flush(&mut self) -> Option<SseEvent> {
        if self.current_data_lines.is_empty() {
            return None;
        }
        Some(SseEvent {
            event: self.current_event.take(),
            data: self.current_data_lines.join("\n"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_data_line_event() {
        let mut parser = SseParser::new();
        let events = parser.feed(b"data: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
        assert_eq!(events[0].event, None);
    }

    #[test]
    fn multi_line_data_joined() {
        let mut parser = SseParser::new();
        let events = parser.feed(b"data: line1\ndata: line2\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2");
    }

    #[test]
    fn event_name_parsed() {
        let mut parser = SseParser::new();
        let events = parser.feed(b"event: message_start\ndata: {}\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.as_deref(), Some("message_start"));
        assert_eq!(events[0].data, "{}");
    }

    #[test]
    fn comment_lines_ignored() {
        let mut parser = SseParser::new();
        let events = parser.feed(b": this is a comment\ndata: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn empty_keepalive_ignored() {
        let mut parser = SseParser::new();
        let events = parser.feed(b"\n\ndata: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn chunk_split_mid_line() {
        let mut parser = SseParser::new();

        let events1 = parser.feed(b"data: hel");
        assert!(events1.is_empty());

        let events2 = parser.feed(b"lo\n\n");
        assert_eq!(events2.len(), 1);
        assert_eq!(events2[0].data, "hello");
    }

    #[test]
    fn chunk_split_mid_newline() {
        let mut parser = SseParser::new();

        let events1 = parser.feed(b"data: hello\n");
        assert!(events1.is_empty()); // no blank line yet

        let events2 = parser.feed(b"\ndata: world\n\n");
        assert_eq!(events2.len(), 2);
        assert_eq!(events2[0].data, "hello");
        assert_eq!(events2[1].data, "world");
    }

    #[test]
    fn multiple_events_in_one_chunk() {
        let mut parser = SseParser::new();
        let events = parser.feed(b"data: first\n\ndata: second\n\n");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "first");
        assert_eq!(events[1].data, "second");
    }

    #[test]
    fn data_without_space_after_colon() {
        let mut parser = SseParser::new();
        let events = parser.feed(b"data:nospace\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "nospace");
    }

    #[test]
    fn crlf_line_endings() {
        let mut parser = SseParser::new();
        let events = parser.feed(b"data: hello\r\n\r\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn flush_emits_remaining() {
        let mut parser = SseParser::new();
        parser.feed(b"data: hello\n");
        // No trailing blank line
        let event = parser.flush();
        assert!(event.is_some());
        assert_eq!(event.unwrap().data, "hello");
    }

    #[test]
    fn flush_nothing_when_empty() {
        let mut parser = SseParser::new();
        assert!(parser.flush().is_none());
    }

    #[test]
    fn done_marker_preserved_as_data() {
        let mut parser = SseParser::new();
        let events = parser.feed(b"data: [DONE]\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "[DONE]");
    }

    #[test]
    fn anthropic_style_event_with_name() {
        let mut parser = SseParser::new();
        let events = parser
            .feed(b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\"}\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.as_deref(), Some("content_block_delta"));
        assert_eq!(events[0].data, "{\"type\":\"content_block_delta\"}");
    }
}
