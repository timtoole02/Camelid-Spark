//! Parse a model's generated text into tool calls (Hybrid Phase 1). The server
//! renders tool definitions through the model's own chat template; this turns the
//! model's *output* back into structured calls. Family-specific: Llama 3.x emits
//! JSON (`{"name":…,"parameters":…}`, optionally `<|python_tag|>`-wrapped);
//! Qwen3/Hermes emit `<tool_call>{…}</tool_call>`. Malformed output yields no
//! calls (the loop then treats the text as a final answer) — never a panic.

use serde_json::Value;

use super::tools::ToolCall;

/// Parse `text` into zero or more tool calls. Empty = no tool call (plain answer).
pub fn parse(text: &str, family: &str) -> Vec<ToolCall> {
    // Ornith / qwen35 emit a custom XML form `<tool_call><function=NAME>
    // <parameter=ARG>VALUE</parameter>…</function></tool_call>` (NOT JSON), so it
    // must be checked BEFORE the qwen/hermes arm (note "qwen35" contains "qwen").
    if family.contains("ornith") || family.contains("qwen35") {
        let calls = parse_ornith(text);
        if !calls.is_empty() {
            return calls;
        }
        // Fall back to hermes/JSON in case a future build emits standard tags.
        let calls = parse_hermes(text);
        if !calls.is_empty() {
            return calls;
        }
        return parse_json(text);
    }
    if family.contains("mistral") {
        let calls = parse_mistral(text);
        if !calls.is_empty() {
            return calls;
        }
        return parse_json(text);
    }
    let hermes_first = family.contains("qwen") || family.contains("hermes");
    if hermes_first {
        let calls = parse_hermes(text);
        if !calls.is_empty() {
            return calls;
        }
        return parse_json(text);
    }
    let calls = parse_json(text);
    if !calls.is_empty() {
        return calls;
    }
    parse_hermes(text)
}

/// `[TOOL_CALLS] [{"name": …, "arguments": {…}}, …]` (Mistral Instruct v0.3+).
fn parse_mistral(text: &str) -> Vec<ToolCall> {
    let marker = "[TOOL_CALLS]";
    if let Some(idx) = text.find(marker) {
        let rest = text[idx + marker.len()..].trim();
        if let Ok(value) = serde_json::from_str::<Value>(rest) {
            return calls_from_value(&value);
        }
        // The model sometimes appends an EOS token or trailing text after the array;
        // try to extract the first balanced [...] substring.
        if let Some(start) = rest.find('[') {
            let slice = &rest[start..];
            if let Ok(value) = serde_json::from_str::<Value>(slice) {
                return calls_from_value(&value);
            }
        }
    }
    // Mistral v0.3 GGUF emits bare JSON arrays without [TOOL_CALLS] marker.
    // Extract the first balanced [...] block, ignoring trailing prose.
    if let Some(arr_slice) = first_json_array(text.trim()) {
        if let Ok(value) = serde_json::from_str::<Value>(arr_slice) {
            let calls = calls_from_value(&value);
            if !calls.is_empty() {
                return calls;
            }
        }
    }
    vec![]
}

/// `<tool_call>{ "name": …, "arguments": { … } }</tool_call>` blocks (Qwen/Hermes).
fn parse_hermes(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        rest = &rest[start + "<tool_call>".len()..];
        let inner = match rest.find("</tool_call>") {
            Some(end) => {
                let inner = &rest[..end];
                rest = &rest[end + "</tool_call>".len()..];
                inner
            }
            None => rest,
        };
        if let Ok(value) = serde_json::from_str::<Value>(inner.trim()) {
            if let Some(call) = call_from_obj(&value) {
                calls.push(call);
            }
        }
    }
    calls
}

/// Ornith / Qwen3.5 custom XML tool calls:
/// `<tool_call>\n<function=NAME>\n<parameter=ARG>\nVALUE\n</parameter>…\n</function>\n</tool_call>`.
/// Parses on the `<function=…>` boundary (the `<tool_call>` wrapper is optional in
/// practice), so a bare function block still lifts. Each `<parameter=ARG>` value keeps
/// the template's wrapper newline stripped; values that look like JSON objects/arrays
/// are decoded (the template `tojson`s mapping/sequence args), scalars stay strings.
fn parse_ornith(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(fstart) = rest.find("<function=") {
        let after = &rest[fstart + "<function=".len()..];
        let Some(name_end) = after.find('>') else {
            break;
        };
        let name = after[..name_end].trim().to_string();
        let body = &after[name_end + 1..];
        let (params_blob, next) = match body.find("</function>") {
            Some(end) => (&body[..end], &body[end + "</function>".len()..]),
            None => (body, ""),
        };

        let mut args = serde_json::Map::new();
        let mut p = params_blob;
        while let Some(ps) = p.find("<parameter=") {
            let pa = &p[ps + "<parameter=".len()..];
            let Some(pname_end) = pa.find('>') else { break };
            let pname = pa[..pname_end].trim().to_string();
            let pbody = &pa[pname_end + 1..];
            let (pval, pnext) = match pbody.find("</parameter>") {
                Some(end) => (&pbody[..end], &pbody[end + "</parameter>".len()..]),
                None => (pbody, ""),
            };
            // The template wraps the value as `>\nVALUE\n</parameter>`; strip exactly
            // one leading + one trailing newline to recover VALUE verbatim.
            let v = pval.strip_prefix('\n').unwrap_or(pval);
            let v = v.strip_suffix('\n').unwrap_or(v);
            let trimmed = v.trim();
            let value = if trimmed.starts_with('{') || trimmed.starts_with('[') {
                serde_json::from_str::<Value>(trimmed)
                    .unwrap_or_else(|_| Value::String(v.to_string()))
            } else {
                Value::String(v.to_string())
            };
            if !pname.is_empty() {
                args.insert(pname, value);
            }
            p = pnext;
        }

        if !name.is_empty() {
            calls.push(ToolCall {
                name,
                args: Value::Object(args),
            });
        }
        rest = next;
    }
    calls
}

/// Bare/`python_tag`-wrapped JSON tool call(s) (Llama 3.x).
fn parse_json(text: &str) -> Vec<ToolCall> {
    let cleaned = strip_markers(text);
    let trimmed = cleaned.trim();
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return calls_from_value(&value);
    }
    // Otherwise try to extract the first balanced {…} object.
    if let Some(slice) = first_json_object(trimmed) {
        if let Ok(value) = serde_json::from_str::<Value>(slice) {
            return calls_from_value(&value);
        }
    }
    Vec::new()
}

fn calls_from_value(value: &Value) -> Vec<ToolCall> {
    match value {
        Value::Array(items) => items.iter().filter_map(call_from_obj).collect(),
        Value::Object(_) => call_from_obj(value).into_iter().collect(),
        _ => Vec::new(),
    }
}

/// Build a call from an object: `name` + args from `parameters`/`arguments`/the
/// object minus the envelope keys. Returns None if there's no usable name.
fn call_from_obj(value: &Value) -> Option<ToolCall> {
    let obj = value.as_object()?;
    // Some models nest under "function": {"name":…,"arguments":…}.
    if let Some(func) = obj.get("function").and_then(Value::as_object) {
        let name = func.get("name").and_then(Value::as_str)?.to_string();
        let args = func
            .get("arguments")
            .or_else(|| func.get("parameters"))
            .cloned()
            .map(coerce_args)
            .unwrap_or_else(|| Value::Object(Default::default()));
        return Some(ToolCall { name, args });
    }
    let name = obj.get("name").and_then(Value::as_str)?.to_string();
    let args = obj
        .get("parameters")
        .or_else(|| obj.get("arguments"))
        .cloned()
        .map(coerce_args)
        .unwrap_or_else(|| {
            let mut rest = obj.clone();
            rest.remove("name");
            rest.remove("type");
            Value::Object(rest)
        });
    Some(ToolCall { name, args })
}

/// Arguments are sometimes a JSON *string* — decode it to an object when so.
fn coerce_args(value: Value) -> Value {
    if let Value::String(s) = &value {
        if let Ok(parsed) = serde_json::from_str::<Value>(s) {
            return parsed;
        }
    }
    value
}

fn strip_markers(text: &str) -> String {
    let mut s = text.to_string();
    for marker in [
        "<|python_tag|>",
        "<|eom_id|>",
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "```json",
        "```",
    ] {
        s = s.replace(marker, " ");
    }
    s
}

/// First balanced `{…}` substring (depth-aware, ignores braces in strings).
fn first_json_object(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{')?;
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// First balanced `[…]` substring (depth-aware, ignores brackets in strings).
fn first_json_array(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let start = bytes.iter().position(|&b| b == b'[')?;
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_llama_json_with_parameters() {
        let out = parse(
            "<|python_tag|>{\"name\": \"read_file\", \"parameters\": {\"path\": \"src/main.rs\"}}<|eom_id|>",
            "llama_bpe_decoder",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[0].args["path"], "src/main.rs");
    }

    #[test]
    fn parses_hermes_qwen_tool_call_tags() {
        let out = parse(
            "sure<tool_call>{\"name\": \"list_dir\", \"arguments\": {\"path\": \".\"}}</tool_call>",
            "qwen3",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "list_dir");
        assert_eq!(out[0].args["path"], ".");
    }

    #[test]
    fn parses_call_embedded_in_prose() {
        let out = parse(
            "I will read it. {\"name\":\"read_file\",\"parameters\":{\"path\":\"a\"}} done",
            "llama",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
    }

    #[test]
    fn plain_answer_yields_no_calls() {
        assert!(parse("The file has 3 lines.", "llama").is_empty());
    }

    #[test]
    fn malformed_json_is_clean_not_a_panic() {
        // Looks like a call but is broken JSON → no calls, no panic.
        assert!(parse("{\"name\": \"read_file\", \"parameters\": {bad", "llama").is_empty());
        assert!(parse("<tool_call>{not json}</tool_call>", "qwen").is_empty());
        // Truncated mid-string and empty input.
        assert!(parse(
            "{\"name\":\"read_file\",\"parameters\":{\"path\":\"no",
            "llama"
        )
        .is_empty());
        assert!(parse("", "llama").is_empty());
    }

    #[test]
    fn double_encoded_args_string_is_normalized_to_object() {
        // Some models emit `parameters`/`arguments` as a JSON-encoded *string*.
        let out = parse(
            "{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"a.txt\\\"}\"}",
            "llama",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].args["path"], "a.txt"); // normalized to a real object
    }

    #[test]
    fn function_envelope_is_unwrapped() {
        // OpenAI-shaped output the model sometimes mirrors back.
        let out = parse(
            "{\"type\":\"function\",\"function\":{\"name\":\"list_dir\",\"arguments\":{\"path\":\".\"}}}",
            "llama",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "list_dir");
        assert_eq!(out[0].args["path"], ".");
    }

    #[test]
    fn multiple_calls_in_one_turn() {
        // Hermes: two tagged calls.
        let hermes = parse(
            "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a\"}}</tool_call>\
             <tool_call>{\"name\":\"list_dir\",\"arguments\":{\"path\":\".\"}}</tool_call>",
            "qwen3",
        );
        assert_eq!(hermes.len(), 2);
        assert_eq!(hermes[0].name, "read_file");
        assert_eq!(hermes[1].name, "list_dir");
        // Llama: a JSON array of calls.
        let arr = parse(
            "[{\"name\":\"read_file\",\"parameters\":{\"path\":\"a\"}},{\"name\":\"search\",\"parameters\":{\"pattern\":\"x\"}}]",
            "llama",
        );
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[1].name, "search");
    }

    #[test]
    fn trailing_and_leading_prose_around_call() {
        let out = parse(
            "Sure, I'll read it now:\n<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a\"}}</tool_call>\nDone.",
            "qwen",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
    }

    #[test]
    fn schema_echo_parses_to_name_with_wrong_args_for_the_gate_to_reject() {
        // The exact 1B failure mode: name is right, args are the schema. The
        // parser must surface it (name parsed) so validate() rejects it with a
        // typed error rather than the parser silently "succeeding".
        let out = parse(
            "{\"name\":\"read_file\",\"parameters\":{\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"],\"type\":\"object\"}}",
            "llama",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
        assert!(out[0].args.get("path").is_none()); // no real value → gate rejects
    }

    #[test]
    fn parses_mistral_tool_calls_marker() {
        let out = parse(
            "[TOOL_CALLS] [{\"name\": \"read_file\", \"arguments\": {\"path\": \"notes.txt\"}}]",
            "mistral",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[0].args["path"], "notes.txt");
    }

    #[test]
    fn parses_mistral_multiple_tool_calls() {
        let out = parse(
            "[TOOL_CALLS] [{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.txt\"}}, {\"name\": \"list_dir\", \"arguments\": {\"path\": \".\"}}]",
            "mistral",
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[0].args["path"], "a.txt");
        assert_eq!(out[1].name, "list_dir");
        assert_eq!(out[1].args["path"], ".");
    }

    #[test]
    fn mistral_falls_back_to_json_without_marker() {
        // If Mistral emits bare JSON (unlikely but possible), the fallback works.
        let out = parse(
            "{\"name\": \"read_file\", \"arguments\": {\"path\": \"x\"}}",
            "mistral",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
    }

    #[test]
    fn mistral_plain_answer_yields_no_calls() {
        assert!(parse("The file contains 3 lines of text.", "mistral").is_empty());
    }

    #[test]
    fn mistral_parses_bare_array_without_marker() {
        let out = parse(
            " [{\"name\": \"read_file\", \"arguments\": {\"path\": \"notes.txt\"}}]\n\nLet me read it.",
            "mistral",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[0].args["path"], "notes.txt");
    }

    #[test]
    fn mistral_parses_bare_multi_call_array() {
        let out = parse(
            "[{\"name\":\"read_file\",\"arguments\":{\"path\":\"a\"}},{\"name\":\"list_dir\",\"arguments\":{\"path\":\".\"}}]\nDone.",
            "mistral",
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[1].name, "list_dir");
    }

    // ---- Ornith / qwen35 custom XML tool-call lift (Bug-1 gate) ----

    /// The exact bytes the Ornith chat template emits for a tool call, routed by the
    /// `qwen35` family (note: "qwen35" contains "qwen", so order matters).
    #[test]
    fn parses_ornith_single_tool_call() {
        let text = "<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n</parameter>\n</function>\n</tool_call>";
        let out = parse(text, "qwen35");
        assert_eq!(out.len(), 1, "exactly one call, single parse");
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[0].args["path"], "notes.txt");
    }

    /// Reasoning must NOT contaminate the tool lift, and a natural-language preamble
    /// before the call (allowed by the template) is ignored.
    #[test]
    fn parses_ornith_call_after_think_and_preamble() {
        let text = "<think>\nI should read the file to count lines.\n</think>\n\nI'll read it now.\n<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n</parameter>\n</function>\n</tool_call>";
        let out = parse(text, "qwen35");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[0].args["path"], "notes.txt");
    }

    /// Multiple parameters; a JSON-object-valued parameter is decoded, a scalar stays
    /// a string. No double-parse.
    #[test]
    fn parses_ornith_multi_param_and_json_value() {
        let text = "<tool_call>\n<function=edit_file>\n<parameter=path>\nsrc/x.rs\n</parameter>\n<parameter=edits>\n{\"a\": 1}\n</parameter>\n</function>\n</tool_call>";
        let out = parse(text, "qwen35");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "edit_file");
        assert_eq!(out[0].args["path"], "src/x.rs");
        assert_eq!(out[0].args["edits"]["a"], 1);
    }

    /// Two calls in one message lift to two structured calls.
    #[test]
    fn parses_ornith_two_calls() {
        let text = "<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=list_dir>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>";
        let out = parse(text, "qwen35");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[0].args["path"], "a.txt");
        assert_eq!(out[1].name, "list_dir");
        assert_eq!(out[1].args["path"], ".");
    }

    /// Plain assistant text (no call) yields no calls — the loop treats it as a final
    /// answer rather than mis-firing a tool.
    #[test]
    fn ornith_plain_answer_no_calls() {
        let text = "<think>\nThe answer is 3.\n</think>\n\nThe file has 3 lines.";
        assert!(parse(text, "qwen35").is_empty());
    }
}
