//! Minimal JSON-object grammar for constrained decoding (`response_format:
//! {"type":"json_object"}`). A byte-level pushdown automaton that decides, at each
//! decode step, which token continuations keep the output a valid JSON-object
//! prefix — so the model can only emit well-formed JSON. Pure and heavily tested;
//! this is the high-value core of the structured-output lane. Arbitrary GBNF and
//! full JSON Schema are deliberate follow-ups.

use std::sync::Arc;

/// The container we are currently inside.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Frame {
    Object,
    Array,
}

/// Sub-state while scanning a JSON number. The "complete" sub-states (a number may
/// validly end here) are `Zero`, `Int`, `Frac`, `Exp`; the rest require more input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Num {
    Sign,     // saw '-', need first digit
    Zero,     // saw leading '0' (complete)
    Int,      // saw 1-9 then digits (complete)
    DotFirst, // saw '.', need first fraction digit
    Frac,     // fraction digits (complete)
    ExpSign,  // saw e/E, may take +/- or a digit
    ExpFirst, // saw e/E and a sign, need first exponent digit
    Exp,      // exponent digits (complete)
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Mode {
    Start,                                    // top level: whitespace then '{'
    Value,                                    // a value is required (after ':')
    Str { escape: bool, hex: u8, key: bool }, // inside a "string"
    Num(Num),                                 // inside a number
    Lit { rest: &'static [u8] },              // matching true / false / null
    Key { allow_close: bool },                // object: a key string (or '}')
    Colon,                                    // object: ':'
    ArrVal { allow_close: bool },             // array: a value (or ']')
    AfterValue,                               // ',' or the container's close
    Done,                                     // top-level object complete
}

/// Incremental JSON-object validator. `advance` consumes one byte and returns an
/// error if that byte cannot extend a valid JSON object. `is_done` is true once the
/// top-level object has fully closed.
#[derive(Clone, Debug)]
pub struct JsonState {
    stack: Vec<Frame>,
    mode: Mode,
}

impl Default for JsonState {
    fn default() -> Self {
        Self::new()
    }
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

fn is_digit(b: u8) -> bool {
    b.is_ascii_digit()
}

fn is_hex(b: u8) -> bool {
    b.is_ascii_hexdigit()
}

/// A byte that can terminate a number (whitespace or a structural close/separator).
fn is_num_delim(b: u8) -> bool {
    is_ws(b) || matches!(b, b',' | b'}' | b']')
}

/// One transition of the shared JSON *number* sub-grammar. Both [`JsonState`] and
/// the schema validator drive numbers through this so the two automata agree
/// byte-for-byte. `integer_only` forbids the fraction/exponent forms (JSON Schema
/// `integer`), keeping constrained output in canonical integer shape.
enum NumStep {
    /// Still inside the number; carries the new sub-state.
    Stay(Num),
    /// A complete number ends here; the caller re-processes `b` as a delimiter.
    Done,
    /// `b` cannot extend the number.
    Reject,
}

fn num_advance(num: Num, b: u8, integer_only: bool) -> NumStep {
    // Complete sub-states end the number on a delimiter (re-processed by the caller).
    let complete = matches!(num, Num::Zero | Num::Int | Num::Frac | Num::Exp);
    if complete && is_num_delim(b) {
        return NumStep::Done;
    }
    let next = match num {
        Num::Sign => match b {
            b'0' => Num::Zero,
            b'1'..=b'9' => Num::Int,
            _ => return NumStep::Reject,
        },
        Num::Zero => match b {
            b'.' if !integer_only => Num::DotFirst,
            b'e' | b'E' if !integer_only => Num::ExpSign,
            _ => return NumStep::Reject, // no leading-zero digits, e.g. "01"
        },
        Num::Int => match b {
            _ if is_digit(b) => Num::Int,
            b'.' if !integer_only => Num::DotFirst,
            b'e' | b'E' if !integer_only => Num::ExpSign,
            _ => return NumStep::Reject,
        },
        Num::DotFirst => {
            if is_digit(b) {
                Num::Frac
            } else {
                return NumStep::Reject;
            }
        }
        Num::Frac => match b {
            _ if is_digit(b) => Num::Frac,
            b'e' | b'E' => Num::ExpSign,
            _ => return NumStep::Reject,
        },
        Num::ExpSign => match b {
            b'+' | b'-' => Num::ExpFirst,
            _ if is_digit(b) => Num::Exp,
            _ => return NumStep::Reject,
        },
        Num::ExpFirst => {
            if is_digit(b) {
                Num::Exp
            } else {
                return NumStep::Reject;
            }
        }
        Num::Exp => {
            if is_digit(b) {
                Num::Exp
            } else {
                return NumStep::Reject;
            }
        }
    };
    NumStep::Stay(next)
}

/// One transition of the shared JSON *string* body sub-grammar (the bytes between
/// the quotes). Shared by [`JsonState`] and the schema validator so both treat
/// escapes, `\u` hex, and raw control characters identically.
///
/// Known relaxation (inherited, not enforced): raw bytes >= 0x80 pass without UTF-8
/// well-formedness checks, and a `\u` escape accepts a lone surrogate. The emitted
/// output is still accepted by essentially all JSON parsers; strict
/// well-formedness / surrogate-pairing is a possible follow-up.
enum StrStep {
    /// Still inside the string; carries the new escape/hex sub-state.
    Stay { escape: bool, hex: u8 },
    /// The closing quote was consumed; the string is complete.
    Close,
    /// `b` cannot extend the string.
    Reject,
}

fn str_advance(escape: bool, hex: u8, b: u8) -> StrStep {
    if hex > 0 {
        if is_hex(b) {
            StrStep::Stay {
                escape: false,
                hex: hex - 1,
            }
        } else {
            StrStep::Reject
        }
    } else if escape {
        match b {
            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => StrStep::Stay {
                escape: false,
                hex: 0,
            },
            b'u' => StrStep::Stay {
                escape: false,
                hex: 4,
            },
            _ => StrStep::Reject,
        }
    } else {
        match b {
            b'"' => StrStep::Close,
            b'\\' => StrStep::Stay {
                escape: true,
                hex: 0,
            },
            // Raw control characters are not allowed in a JSON string.
            0x00..=0x1F => StrStep::Reject,
            _ => StrStep::Stay {
                escape: false,
                hex: 0,
            },
        }
    }
}

impl JsonState {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            mode: Mode::Start,
        }
    }

    /// True when a complete top-level JSON object has been emitted (the model may
    /// stop). The decode loop stops as soon as this becomes true.
    pub fn is_done(&self) -> bool {
        self.mode == Mode::Done
    }

    /// Would `bytes` keep the output a valid JSON-object prefix from the current
    /// state? Used to mask a candidate token without mutating the live state.
    pub fn accepts(&self, bytes: &[u8]) -> bool {
        let mut probe = self.clone();
        for &b in bytes {
            if probe.advance(b).is_err() {
                return false;
            }
        }
        true
    }

    /// Begin a value on `b` (used after ':' and after array '[' / ','). Strings
    /// started here are values (not keys).
    fn start_value(&mut self, b: u8) -> Result<(), ()> {
        self.mode = match b {
            b'{' => {
                self.stack.push(Frame::Object);
                Mode::Key { allow_close: true }
            }
            b'[' => {
                self.stack.push(Frame::Array);
                Mode::ArrVal { allow_close: true }
            }
            b'"' => Mode::Str {
                escape: false,
                hex: 0,
                key: false,
            },
            b'-' => Mode::Num(Num::Sign),
            b'0' => Mode::Num(Num::Zero),
            b'1'..=b'9' => Mode::Num(Num::Int),
            b't' => Mode::Lit { rest: b"rue" },
            b'f' => Mode::Lit { rest: b"alse" },
            b'n' => Mode::Lit { rest: b"ull" },
            _ => return Err(()),
        };
        Ok(())
    }

    /// A container (or the top-level object) just closed, completing a value.
    fn after_close(&mut self) {
        self.mode = if self.stack.is_empty() {
            Mode::Done
        } else {
            Mode::AfterValue
        };
    }

    /// Consume one byte; `Err(())` means the byte cannot extend a valid JSON object.
    /// The error is intentionally information-free — a rejected byte is a rejected
    /// byte; the caller only needs the yes/no.
    #[allow(clippy::result_unit_err)]
    pub fn advance(&mut self, b: u8) -> Result<(), ()> {
        match self.mode {
            Mode::Start => {
                if is_ws(b) {
                    Ok(())
                } else if b == b'{' {
                    self.stack.push(Frame::Object);
                    self.mode = Mode::Key { allow_close: true };
                    Ok(())
                } else {
                    Err(())
                }
            }
            Mode::Value => {
                if is_ws(b) {
                    Ok(())
                } else {
                    self.start_value(b)
                }
            }
            Mode::ArrVal { allow_close } => {
                if is_ws(b) {
                    Ok(())
                } else if allow_close && b == b']' {
                    self.stack.pop();
                    self.after_close();
                    Ok(())
                } else {
                    self.start_value(b)
                }
            }
            Mode::Key { allow_close } => {
                if is_ws(b) {
                    Ok(())
                } else if allow_close && b == b'}' {
                    self.stack.pop();
                    self.after_close();
                    Ok(())
                } else if b == b'"' {
                    self.mode = Mode::Str {
                        escape: false,
                        hex: 0,
                        key: true,
                    };
                    Ok(())
                } else {
                    Err(())
                }
            }
            Mode::Colon => {
                if is_ws(b) {
                    Ok(())
                } else if b == b':' {
                    self.mode = Mode::Value;
                    Ok(())
                } else {
                    Err(())
                }
            }
            Mode::Str { escape, hex, key } => match str_advance(escape, hex, b) {
                StrStep::Stay { escape, hex } => {
                    self.mode = Mode::Str { escape, hex, key };
                    Ok(())
                }
                StrStep::Close => {
                    self.mode = if key { Mode::Colon } else { Mode::AfterValue };
                    Ok(())
                }
                StrStep::Reject => Err(()),
            },
            Mode::Lit { rest } => match rest.split_first() {
                Some((&first, tail)) if first == b => {
                    if tail.is_empty() {
                        self.mode = Mode::AfterValue;
                    } else {
                        self.mode = Mode::Lit { rest: tail };
                    }
                    Ok(())
                }
                _ => Err(()),
            },
            Mode::Num(num) => self.advance_num(num, b),
            Mode::AfterValue => {
                if is_ws(b) {
                    return Ok(());
                }
                match self.stack.last().copied() {
                    Some(Frame::Object) => match b {
                        b',' => {
                            self.mode = Mode::Key { allow_close: false };
                            Ok(())
                        }
                        b'}' => {
                            self.stack.pop();
                            self.after_close();
                            Ok(())
                        }
                        _ => Err(()),
                    },
                    Some(Frame::Array) => match b {
                        b',' => {
                            self.mode = Mode::ArrVal { allow_close: false };
                            Ok(())
                        }
                        b']' => {
                            self.stack.pop();
                            self.after_close();
                            Ok(())
                        }
                        _ => Err(()),
                    },
                    None => Err(()),
                }
            }
            Mode::Done => {
                // After the top-level object, only trailing whitespace is valid.
                if is_ws(b) {
                    Ok(())
                } else {
                    Err(())
                }
            }
        }
    }

    fn advance_num(&mut self, num: Num, b: u8) -> Result<(), ()> {
        match num_advance(num, b, false) {
            NumStep::Done => {
                // The delimiter that ended the number is re-processed by `AfterValue`.
                self.mode = Mode::AfterValue;
                self.advance(b)
            }
            NumStep::Stay(next) => {
                self.mode = Mode::Num(next);
                Ok(())
            }
            NumStep::Reject => Err(()),
        }
    }
}

// Compile-time caps on schema dimensions. Every declared property and enum
// member multiplies the per-step full-vocab `accepts()` scan, and `SchemaState`
// clones (including one `used: Vec<bool>` per object frame) once per candidate
// token per decode step — an uncapped schema turns one request into gigabytes
// of memcpy per step on the single engine worker. Oversized schemas are a
// typed 400 at compile time, never a slow decode.

/// Maximum declared properties per object.
const MAX_OBJECT_PROPERTIES: usize = 256;
/// Maximum bytes in a declared property name.
const MAX_PROPERTY_NAME_BYTES: usize = 64;
/// Maximum members in a string `enum`.
const MAX_ENUM_MEMBERS: usize = 256;
/// Maximum bytes of one `enum`/`const` member's JSON encoding (quotes included).
const MAX_ENUM_MEMBER_BYTES: usize = 256;
/// Maximum nesting depth of the schema tree (the root node is depth 0).
/// serde_json's parse recursion limit bounds this in practice today; the cap
/// makes the bound explicit and owned by this subsystem.
const MAX_SCHEMA_DEPTH: usize = 32;

/// Failure compiling a JSON Schema into the supported subset. The message names
/// the offending keyword/type so the API can return a precise, honest 400 rather
/// than silently ignoring a constraint it cannot enforce.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchemaError(String);

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SchemaError {}

fn serr(msg: impl Into<String>) -> SchemaError {
    SchemaError(msg.into())
}

/// A compiled node of the supported JSON Schema subset (see `compile_root` for the
/// exact boundary). Non-container values only ever appear inside a container, so
/// their completion is always disambiguated by the enclosing `,`/`}`/`]`.
#[derive(Clone, Debug)]
enum Schema {
    Str,
    Integer,
    Number,
    Bool,
    Null,
    /// String `enum`/`const`: the value must equal one of these canonical JSON
    /// encodings (e.g. `"celsius"`).
    Enum(Arc<Vec<Vec<u8>>>),
    Object(Arc<ObjectSchema>),
    Array(Arc<Schema>),
    /// A `type` union whose members start with distinct bytes (e.g. the OpenAI
    /// nullable pattern `["string","null"]`); the first value byte selects the member.
    Union(Arc<Vec<Schema>>),
}

#[derive(Debug)]
struct ObjectSchema {
    props: Vec<PropSchema>,
}

#[derive(Debug)]
struct PropSchema {
    name: String,
    schema: Schema,
    required: bool,
}

/// Keywords that are pure annotations — safe to ignore on any node.
const IGNORED_KEYWORDS: &[&str] = &[
    "description",
    "title",
    "default",
    "examples",
    "$schema",
    "$id",
    "$comment",
    "readOnly",
    "writeOnly",
    "deprecated",
    "$defs",
    "definitions",
];

/// Reject any keyword that is neither expected for this node nor a known
/// annotation. This is what makes the subset *fail-closed*: an unrecognized or
/// unenforceable keyword is an error, never a silently dropped constraint.
fn reject_unknown(
    map: &serde_json::Map<String, serde_json::Value>,
    allowed: &[&str],
) -> Result<(), SchemaError> {
    for key in map.keys() {
        let k = key.as_str();
        if !allowed.contains(&k) && !IGNORED_KEYWORDS.contains(&k) {
            return Err(serr(format!("unsupported schema keyword: {k}")));
        }
    }
    Ok(())
}

/// Compile a root schema. The root must be an object or array so that completion
/// (`is_done`) is triggered by a unique closing `}`/`]`; top-level scalars have no
/// terminator and are rejected.
fn compile_root(schema: &serde_json::Value) -> Result<Schema, SchemaError> {
    let compiled = compile_node(schema, 0)?;
    match compiled {
        Schema::Object(_) | Schema::Array(_) => Ok(compiled),
        _ => Err(serr("the root schema must be an object or array")),
    }
}

fn compile_node(schema: &serde_json::Value, depth: usize) -> Result<Schema, SchemaError> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(serr(format!(
            "schema nesting exceeds the supported maximum depth of {MAX_SCHEMA_DEPTH}"
        )));
    }
    let map = match schema {
        serde_json::Value::Object(map) => map,
        _ => return Err(serr("each schema must be a JSON object")),
    };
    if let Some(constant) = map.get("const") {
        reject_unknown(map, &["const", "type"])?;
        reject_conflicting_literal_type(map, "const")?;
        return compile_string_literals(std::slice::from_ref(constant), "const");
    }
    if let Some(values) = map.get("enum") {
        reject_unknown(map, &["enum", "type"])?;
        reject_conflicting_literal_type(map, "enum")?;
        let arr = values
            .as_array()
            .ok_or_else(|| serr("`enum` must be an array"))?;
        return compile_string_literals(arr, "enum");
    }
    let types: Vec<&str> = match map.get("type") {
        None => {
            return Err(serr(
                "schema must declare a `type` (untyped/any schemas are not supported yet)",
            ))
        }
        Some(serde_json::Value::String(s)) => vec![s.as_str()],
        Some(serde_json::Value::Array(items)) => {
            if items.is_empty() {
                return Err(serr("`type` array must be non-empty"));
            }
            items
                .iter()
                .map(|v| {
                    v.as_str()
                        .ok_or_else(|| serr("`type` array entries must be strings"))
                })
                .collect::<Result<Vec<_>, _>>()?
        }
        Some(_) => return Err(serr("`type` must be a string or an array of type strings")),
    };
    // Validate keywords once against the union of the member types' allowed keys, so a
    // nullable object like {"type":["object","null"], "properties":{...}} keeps its
    // object keywords without the `null` member rejecting them.
    let mut allowed: Vec<&str> = Vec::new();
    for &ty in &types {
        for &key in type_allowed_keys(ty) {
            if !allowed.contains(&key) {
                allowed.push(key);
            }
        }
    }
    reject_unknown(map, &allowed)?;
    if types.len() == 1 {
        return compile_typed(types[0], map, depth);
    }
    // A multi-type union (e.g. the OpenAI nullable pattern ["string","null"]): compile
    // each member reusing the surrounding keywords, and require distinct start bytes so
    // a single lookahead selects the branch.
    let mut members = Vec::with_capacity(types.len());
    for &ty in &types {
        members.push(compile_typed(ty, map, depth)?);
    }
    ensure_disjoint_start_bytes(&members)?;
    Ok(Schema::Union(Arc::new(members)))
}

/// Compile a single JSON `type` (with its surrounding keywords) into a node. Keyword
/// validation is the caller's responsibility (done once over the union of members).
fn compile_typed(
    ty: &str,
    map: &serde_json::Map<String, serde_json::Value>,
    depth: usize,
) -> Result<Schema, SchemaError> {
    match ty {
        "object" => compile_object(map, depth),
        "array" => compile_array(map, depth),
        "string" => Ok(Schema::Str),
        "integer" => Ok(Schema::Integer),
        "number" => Ok(Schema::Number),
        "boolean" => Ok(Schema::Bool),
        "null" => Ok(Schema::Null),
        other => Err(serr(format!("unsupported `type`: {other}"))),
    }
}

/// The keywords a given type is allowed to carry (beyond annotations).
fn type_allowed_keys(ty: &str) -> &'static [&'static str] {
    match ty {
        "object" => &["type", "properties", "required", "additionalProperties"],
        "array" => &["type", "items"],
        _ => &["type"],
    }
}

/// The distinct bytes a value matching `schema` may begin with.
fn first_bytes(schema: &Schema) -> Vec<u8> {
    match schema {
        Schema::Str => vec![b'"'],
        Schema::Enum(cands) => cands.iter().filter_map(|c| c.first().copied()).collect(),
        Schema::Integer | Schema::Number => {
            let mut bytes = vec![b'-'];
            bytes.extend(b'0'..=b'9');
            bytes
        }
        Schema::Bool => vec![b't', b'f'],
        Schema::Null => vec![b'n'],
        Schema::Object(_) => vec![b'{'],
        Schema::Array(_) => vec![b'['],
        Schema::Union(members) => members.iter().flat_map(first_bytes).collect(),
    }
}

/// A `type` union is supported only when its members start with disjoint bytes, so a
/// single lookahead byte selects the branch (true for every `["T","null"]` nullable
/// union). Overlapping shapes (e.g. integer + number) are rejected fail-closed.
fn ensure_disjoint_start_bytes(members: &[Schema]) -> Result<(), SchemaError> {
    let mut seen = [false; 256];
    for member in members {
        for b in first_bytes(member) {
            if seen[b as usize] {
                return Err(serr(
                    "ambiguous `type` union: members share a starting byte; only unions of distinct value shapes (e.g. [\"string\",\"null\"]) are supported",
                ));
            }
            seen[b as usize] = true;
        }
    }
    Ok(())
}

/// String `const`/`enum` may carry a redundant `type`, but only a string one: a
/// non-string `type` alongside string members is a self-contradictory schema, which
/// we reject rather than silently ignore (the members would be enforced either way).
fn reject_conflicting_literal_type(
    map: &serde_json::Map<String, serde_json::Value>,
    keyword: &str,
) -> Result<(), SchemaError> {
    match map.get("type") {
        None => Ok(()),
        Some(serde_json::Value::String(s)) if s == "string" => Ok(()),
        Some(_) => Err(serr(format!(
            "`{keyword}` members are strings; a non-string `type` alongside them is contradictory"
        ))),
    }
}

fn compile_string_literals(
    values: &[serde_json::Value],
    keyword: &str,
) -> Result<Schema, SchemaError> {
    if values.is_empty() {
        return Err(serr(format!("`{keyword}` must be non-empty")));
    }
    if values.len() > MAX_ENUM_MEMBERS {
        return Err(serr(format!(
            "`{keyword}` declares {} members; the supported maximum is {MAX_ENUM_MEMBERS}",
            values.len()
        )));
    }
    let mut encodings = Vec::with_capacity(values.len());
    for value in values {
        match value {
            serde_json::Value::String(s) => {
                // Constrained decoding matches the value's exact UTF-8 bytes, but the
                // per-token byte table (`grammar_token_bytes`) decodes each token in
                // isolation and drops incomplete UTF-8: a byte-level-BPE / byte-fallback
                // tokenizer emits a multibyte character as byte fragments, each of which
                // decodes alone to "" and is masked out. A non-ASCII literal could
                // therefore never be produced and the automaton would dead-end
                // mid-generation. Reject it here so an unenforceable literal is an honest
                // request-time 400, not a constraint that compiles and then fails closed
                // in the middle of decoding.
                if !s.is_ascii() {
                    return Err(serr(format!(
                        "`{keyword}` member {s:?} contains non-ASCII characters; constrained string values are ASCII-only"
                    )));
                }
                let encoded = serde_json::to_vec(value)
                    .map_err(|e| serr(format!("`{keyword}` member is not serializable: {e}")))?;
                if encoded.len() > MAX_ENUM_MEMBER_BYTES {
                    return Err(serr(format!(
                        "`{keyword}` member {s:?} encodes to {} bytes; the supported maximum is {MAX_ENUM_MEMBER_BYTES}",
                        encoded.len()
                    )));
                }
                encodings.push(encoded);
            }
            _ => {
                return Err(serr(format!(
                    "only string `{keyword}` members are supported yet"
                )))
            }
        }
    }
    Ok(Schema::Enum(Arc::new(encodings)))
}

fn compile_object(
    map: &serde_json::Map<String, serde_json::Value>,
    depth: usize,
) -> Result<Schema, SchemaError> {
    match map.get("additionalProperties") {
        Some(serde_json::Value::Bool(false)) => {}
        Some(serde_json::Value::Bool(true)) | None => {
            return Err(serr(
                "objects must set additionalProperties:false (open objects are not supported yet)",
            ))
        }
        Some(_) => {
            return Err(serr(
                "additionalProperties must be false (schema-valued additionalProperties is not supported yet)",
            ))
        }
    }
    let required: Vec<&str> = match map.get("required") {
        None => Vec::new(),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .ok_or_else(|| serr("`required` entries must be strings"))
            })
            .collect::<Result<_, _>>()?,
        Some(_) => return Err(serr("`required` must be an array")),
    };
    let required_set: std::collections::HashSet<&str> = required.iter().copied().collect();
    let mut props = Vec::new();
    if let Some(properties) = map.get("properties") {
        let properties = properties
            .as_object()
            .ok_or_else(|| serr("`properties` must be an object"))?;
        if properties.len() > MAX_OBJECT_PROPERTIES {
            return Err(serr(format!(
                "`properties` declares {} properties; the supported maximum is {MAX_OBJECT_PROPERTIES}",
                properties.len()
            )));
        }
        for (name, sub) in properties {
            if name.len() > MAX_PROPERTY_NAME_BYTES {
                return Err(serr(format!(
                    "property name {name:?} is {} bytes; the supported maximum is {MAX_PROPERTY_NAME_BYTES}",
                    name.len()
                )));
            }
            if !is_simple_key(name) {
                return Err(serr(format!(
                    "property name {name:?} requires JSON escaping; not supported yet"
                )));
            }
            props.push(PropSchema {
                name: name.clone(),
                schema: compile_node(sub, depth + 1)?,
                required: required_set.contains(name.as_str()),
            });
        }
    }
    let declared: std::collections::HashSet<&str> = props.iter().map(|p| p.name.as_str()).collect();
    for name in &required {
        if !declared.contains(name) {
            return Err(serr(format!(
                "required property {name:?} is not declared in `properties`"
            )));
        }
    }
    Ok(Schema::Object(Arc::new(ObjectSchema { props })))
}

fn compile_array(
    map: &serde_json::Map<String, serde_json::Value>,
    depth: usize,
) -> Result<Schema, SchemaError> {
    match map.get("items") {
        Some(items) => Ok(Schema::Array(Arc::new(compile_node(items, depth + 1)?))),
        None => Err(serr(
            "array schemas must declare `items` (unconstrained arrays are not supported yet)",
        )),
    }
}

/// A property name we can match with a plain byte trie: no characters that JSON
/// would have to escape (`"`, `\`, or control chars). Constrained decoding then
/// emits keys in this canonical unescaped form.
fn is_simple_key(name: &str) -> bool {
    name.bytes().all(|b| b >= 0x20 && b != b'"' && b != b'\\')
}

/// A container we are currently inside, with the schema state needed to validate
/// its members.
#[derive(Clone, Debug)]
enum SchemaFrame {
    Object {
        schema: Arc<ObjectSchema>,
        used: Vec<bool>,
    },
    Array {
        items: Schema,
    },
}

#[derive(Clone, Debug)]
enum SchemaMode {
    /// Leading whitespace then a value matching this schema.
    Value(Schema),
    /// Inside an array: whitespace, a value matching the element schema, or `]`.
    ArrayElem { allow_close: bool },
    /// Inside an object: whitespace, a `"` key, or `}`.
    Key { allow_close: bool },
    /// Scanning an object key, constrained to the declared property names.
    KeyStr { matched: Vec<u8> },
    /// Object: `:` then a value matching the resolved property's schema.
    Colon(Schema),
    /// Scanning a string value body.
    Str { escape: bool, hex: u8 },
    /// Scanning a number/integer value body.
    Num { st: Num, integer: bool },
    /// Matching a `true`/`false`/`null` literal tail.
    Lit { rest: &'static [u8] },
    /// Matching a string `enum`/`const`: `viable` are the still-possible candidate
    /// indices, `pos` the number of bytes matched so far.
    Enum {
        cands: Arc<Vec<Vec<u8>>>,
        viable: Vec<usize>,
        pos: usize,
    },
    /// A value just completed inside a container: `,` or the container's close.
    AfterValue,
    /// The root value is complete; only trailing whitespace remains.
    Done,
    /// A byte was rejected: the state is poisoned. Accepts nothing and is never
    /// done, so a rejected `advance` can only surface as an error — never as a
    /// silently "complete" value (`advance` parks the live mode in `Done` while
    /// dispatching, so without this sink every rejection would fail open).
    Failed,
}

/// Incremental validator for the supported JSON Schema subset. Mirrors [`JsonState`]
/// (same `accepts`/`advance`/`is_done` contract) but every step is directed by the
/// schema, so only bytes that keep the output a valid prefix of a schema-matching
/// value are accepted.
#[derive(Clone, Debug)]
pub struct SchemaState {
    stack: Vec<SchemaFrame>,
    mode: SchemaMode,
}

impl SchemaState {
    fn new(schema: Schema) -> Self {
        Self {
            stack: Vec::new(),
            mode: SchemaMode::Value(schema),
        }
    }

    fn is_done(&self) -> bool {
        matches!(self.mode, SchemaMode::Done)
    }

    /// Would appending `bytes` keep the output a valid prefix of the constrained
    /// value? Probes on a clone so the live state is untouched.
    ///
    /// Cost note: the decode loop calls this once per vocab token per step, so each
    /// step clones this state O(vocab) times (as the existing `json_object` path
    /// already does). Acceptable for opt-in constrained decoding; if structured
    /// output becomes hot, a reusable scratch state reset from `self` (instead of a
    /// clone) would remove the per-token allocation.
    fn accepts(&self, bytes: &[u8]) -> bool {
        let mut probe = self.clone();
        for &b in bytes {
            if probe.advance(b).is_err() {
                return false;
            }
        }
        true
    }

    fn advance(&mut self, b: u8) -> Result<(), ()> {
        let result = self.advance_inner(b);
        if result.is_err() {
            // Fail sticky-closed: `advance_inner` parks the live mode in `Done`
            // while it dispatches, so an error path would otherwise leave the
            // state reporting `is_done()` — and a truncated/invalid value would
            // finish as a clean "stop". Poison the state instead.
            self.mode = SchemaMode::Failed;
        }
        result
    }

    fn advance_inner(&mut self, b: u8) -> Result<(), ()> {
        match std::mem::replace(&mut self.mode, SchemaMode::Done) {
            SchemaMode::Value(schema) => {
                if is_ws(b) {
                    self.mode = SchemaMode::Value(schema);
                    Ok(())
                } else {
                    self.start_value(schema, b)
                }
            }
            SchemaMode::ArrayElem { allow_close } => {
                if is_ws(b) {
                    self.mode = SchemaMode::ArrayElem { allow_close };
                    Ok(())
                } else if allow_close && b == b']' {
                    self.pop_close();
                    Ok(())
                } else {
                    let items = match self.stack.last() {
                        Some(SchemaFrame::Array { items }) => items.clone(),
                        _ => return Err(()),
                    };
                    self.start_value(items, b)
                }
            }
            SchemaMode::Key { allow_close } => {
                if is_ws(b) {
                    self.mode = SchemaMode::Key { allow_close };
                    Ok(())
                } else if allow_close && b == b'}' {
                    if self.required_satisfied() {
                        self.pop_close();
                        Ok(())
                    } else {
                        Err(())
                    }
                } else if b == b'"' && self.has_unused_property() {
                    self.mode = SchemaMode::KeyStr {
                        matched: Vec::new(),
                    };
                    Ok(())
                } else {
                    Err(())
                }
            }
            SchemaMode::KeyStr { matched } => self.advance_key(matched, b),
            SchemaMode::Colon(schema) => {
                if is_ws(b) {
                    self.mode = SchemaMode::Colon(schema);
                    Ok(())
                } else if b == b':' {
                    self.mode = SchemaMode::Value(schema);
                    Ok(())
                } else {
                    Err(())
                }
            }
            SchemaMode::Str { escape, hex } => match str_advance(escape, hex, b) {
                StrStep::Stay { escape, hex } => {
                    self.mode = SchemaMode::Str { escape, hex };
                    Ok(())
                }
                StrStep::Close => {
                    self.after_value();
                    Ok(())
                }
                StrStep::Reject => Err(()),
            },
            SchemaMode::Num { st, integer } => match num_advance(st, b, integer) {
                NumStep::Stay(next) => {
                    self.mode = SchemaMode::Num { st: next, integer };
                    Ok(())
                }
                NumStep::Done => {
                    // The delimiter that ended the number is re-processed below.
                    self.after_value();
                    self.advance(b)
                }
                NumStep::Reject => Err(()),
            },
            SchemaMode::Lit { rest } => match rest.split_first() {
                Some((&first, tail)) if first == b => {
                    if tail.is_empty() {
                        self.after_value();
                    } else {
                        self.mode = SchemaMode::Lit { rest: tail };
                    }
                    Ok(())
                }
                _ => Err(()),
            },
            SchemaMode::Enum { cands, viable, pos } => self.advance_enum(cands, viable, pos, b),
            SchemaMode::AfterValue => self.advance_after_value(b),
            SchemaMode::Done => {
                if is_ws(b) {
                    self.mode = SchemaMode::Done;
                    Ok(())
                } else {
                    Err(())
                }
            }
            SchemaMode::Failed => Err(()),
        }
    }

    /// Begin a value of `schema` on its first non-whitespace byte.
    fn start_value(&mut self, schema: Schema, b: u8) -> Result<(), ()> {
        match schema {
            Schema::Object(os) => {
                if b == b'{' {
                    let used = vec![false; os.props.len()];
                    self.stack.push(SchemaFrame::Object { schema: os, used });
                    self.mode = SchemaMode::Key { allow_close: true };
                    Ok(())
                } else {
                    Err(())
                }
            }
            Schema::Array(items) => {
                if b == b'[' {
                    self.stack.push(SchemaFrame::Array {
                        items: (*items).clone(),
                    });
                    self.mode = SchemaMode::ArrayElem { allow_close: true };
                    Ok(())
                } else {
                    Err(())
                }
            }
            Schema::Str => {
                if b == b'"' {
                    self.mode = SchemaMode::Str {
                        escape: false,
                        hex: 0,
                    };
                    Ok(())
                } else {
                    Err(())
                }
            }
            Schema::Integer => self.start_number(b, true),
            Schema::Number => self.start_number(b, false),
            Schema::Bool => match b {
                b't' => {
                    self.mode = SchemaMode::Lit { rest: b"rue" };
                    Ok(())
                }
                b'f' => {
                    self.mode = SchemaMode::Lit { rest: b"alse" };
                    Ok(())
                }
                _ => Err(()),
            },
            Schema::Null => {
                if b == b'n' {
                    self.mode = SchemaMode::Lit { rest: b"ull" };
                    Ok(())
                } else {
                    Err(())
                }
            }
            Schema::Enum(cands) => {
                let viable: Vec<usize> = cands
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.first() == Some(&b))
                    .map(|(i, _)| i)
                    .collect();
                if viable.is_empty() {
                    return Err(());
                }
                let pos = 1;
                if viable.iter().any(|&i| cands[i].len() == pos) {
                    self.after_value();
                } else {
                    self.mode = SchemaMode::Enum { cands, viable, pos };
                }
                Ok(())
            }
            Schema::Union(members) => {
                // Distinct start bytes are guaranteed at compile time, so at most one
                // member can begin with `b`. `start_value` is side-effect-free on Err,
                // so trying members in turn is safe.
                for member in members.iter() {
                    if self.start_value(member.clone(), b).is_ok() {
                        return Ok(());
                    }
                }
                Err(())
            }
        }
    }

    fn start_number(&mut self, b: u8, integer: bool) -> Result<(), ()> {
        let st = match b {
            b'-' => Num::Sign,
            b'0' => Num::Zero,
            b'1'..=b'9' => Num::Int,
            _ => return Err(()),
        };
        self.mode = SchemaMode::Num { st, integer };
        Ok(())
    }

    /// Resolve an object key. During scanning (`b` is a content byte) the key is
    /// constrained to remain a prefix of some not-yet-used declared property; on the
    /// closing quote it must equal exactly one such property, which is then marked
    /// used (so duplicate keys are rejected) and whose schema drives the value.
    fn advance_key(&mut self, mut matched: Vec<u8>, b: u8) -> Result<(), ()> {
        let schema = match self.stack.last() {
            Some(SchemaFrame::Object { schema, .. }) => schema.clone(),
            _ => return Err(()),
        };
        if b == b'"' {
            let Some(i) = schema
                .props
                .iter()
                .position(|p| p.name.as_bytes() == matched.as_slice())
            else {
                return Err(());
            };
            match self.stack.last_mut() {
                Some(SchemaFrame::Object { used, .. }) if !used[i] => used[i] = true,
                _ => return Err(()),
            }
            self.mode = SchemaMode::Colon(schema.props[i].schema.clone());
            Ok(())
        } else {
            if b < 0x20 || b == b'\\' {
                return Err(());
            }
            matched.push(b);
            let used = match self.stack.last() {
                Some(SchemaFrame::Object { used, .. }) => used,
                _ => return Err(()),
            };
            let feasible = schema
                .props
                .iter()
                .enumerate()
                .any(|(i, p)| !used[i] && p.name.as_bytes().starts_with(&matched));
            if feasible {
                self.mode = SchemaMode::KeyStr { matched };
                Ok(())
            } else {
                Err(())
            }
        }
    }

    fn advance_enum(
        &mut self,
        cands: Arc<Vec<Vec<u8>>>,
        viable: Vec<usize>,
        pos: usize,
        b: u8,
    ) -> Result<(), ()> {
        let next: Vec<usize> = viable
            .into_iter()
            .filter(|&i| cands[i].get(pos) == Some(&b))
            .collect();
        if next.is_empty() {
            return Err(());
        }
        let npos = pos + 1;
        if next.iter().any(|&i| cands[i].len() == npos) {
            // String enum encodings are prefix-free, so a fully matched candidate
            // cannot be extended: the value is complete.
            self.after_value();
        } else {
            self.mode = SchemaMode::Enum {
                cands,
                viable: next,
                pos: npos,
            };
        }
        Ok(())
    }

    /// A value just completed inside a container.
    fn after_value(&mut self) {
        self.mode = SchemaMode::AfterValue;
    }

    /// After a value: a `,` continues the container or its close finishes it.
    fn advance_after_value(&mut self, b: u8) -> Result<(), ()> {
        if is_ws(b) {
            self.mode = SchemaMode::AfterValue;
            return Ok(());
        }
        let is_object = match self.stack.last() {
            Some(SchemaFrame::Object { .. }) => true,
            Some(SchemaFrame::Array { .. }) => false,
            None => return Err(()),
        };
        if is_object {
            match b {
                b',' => {
                    if self.has_unused_property() {
                        self.mode = SchemaMode::Key { allow_close: false };
                        Ok(())
                    } else {
                        Err(())
                    }
                }
                b'}' => {
                    if self.required_satisfied() {
                        self.pop_close();
                        Ok(())
                    } else {
                        Err(())
                    }
                }
                _ => Err(()),
            }
        } else {
            match b {
                b',' => {
                    self.mode = SchemaMode::ArrayElem { allow_close: false };
                    Ok(())
                }
                b']' => {
                    self.pop_close();
                    Ok(())
                }
                _ => Err(()),
            }
        }
    }

    /// Pop the just-closed container; the root closing finishes the whole value.
    fn pop_close(&mut self) {
        self.stack.pop();
        self.mode = if self.stack.is_empty() {
            SchemaMode::Done
        } else {
            SchemaMode::AfterValue
        };
    }

    /// Whether the current object still has a declared property left to emit.
    /// Gates `,` after a member and the `"` that opens a key: with
    /// `additionalProperties:false`, once every declared property is used no key
    /// could follow, so allowing `,`/`"` would walk into a dead end (the common
    /// BPE token `,"` would pass the mask and leave the next step with an
    /// all-false mask). This gate can never lock an object out: no unused
    /// properties implies every required property is used, i.e.
    /// `required_satisfied()`, so `}` is legal at that point — in `AfterValue`
    /// and in `Key { allow_close: true }` alike.
    fn has_unused_property(&self) -> bool {
        match self.stack.last() {
            Some(SchemaFrame::Object { used, .. }) => used.iter().any(|u| !u),
            _ => false,
        }
    }

    /// Whether every required property of the current object has been emitted.
    fn required_satisfied(&self) -> bool {
        match self.stack.last() {
            Some(SchemaFrame::Object { schema, used }) => schema
                .props
                .iter()
                .enumerate()
                .all(|(i, p)| !p.required || used[i]),
            _ => false,
        }
    }
}

/// Dispatch across the constrained-decoding backends the decode loop can drive.
///
/// The loop needs only three operations — [`ConstraintState::accepts`] to mask a
/// candidate token, [`ConstraintState::advance`] to commit the chosen token's
/// bytes, and [`ConstraintState::is_done`] to know when the constrained value is
/// complete — so every backend exposes exactly that surface and the loop never has
/// to know which one is active. Today the only backend is the JSON-object grammar;
/// JSON Schema and GBNF are follow-ups that add variants here without touching the
/// decode loop.
#[derive(Clone, Debug)]
pub enum ConstraintState {
    /// `response_format: {"type":"json_object"}` — any valid JSON object.
    Json(JsonState),
    /// `response_format: {"type":"json_schema", ...}` — a value matching a compiled
    /// JSON Schema (the supported subset; see [`ConstraintSpec::from_schema`]).
    Schema(SchemaState),
}

impl ConstraintState {
    /// Construct the JSON-object constraint (the `json_object` response format).
    pub fn new_json() -> Self {
        Self::Json(JsonState::new())
    }

    /// Would appending `bytes` keep the output a valid prefix of the constrained
    /// value? Used to mask a candidate token without mutating the live state.
    pub fn accepts(&self, bytes: &[u8]) -> bool {
        match self {
            Self::Json(state) => state.accepts(bytes),
            Self::Schema(state) => state.accepts(bytes),
        }
    }

    /// Commit one byte of the chosen token. `Err(())` means the byte cannot extend
    /// the constrained value; the per-step mask guarantees this never happens for a
    /// token the loop actually selected.
    #[allow(clippy::result_unit_err)]
    pub fn advance(&mut self, b: u8) -> Result<(), ()> {
        match self {
            Self::Json(state) => state.advance(b),
            Self::Schema(state) => state.advance(b),
        }
    }

    /// True once the constrained value is complete (the model may stop).
    pub fn is_done(&self) -> bool {
        match self {
            Self::Json(state) => state.is_done(),
            Self::Schema(state) => state.is_done(),
        }
    }
}

/// A validated, cheaply-clonable description of the active output constraint.
///
/// Built once at request time — so a schema outside the supported subset surfaces
/// as an error before generation starts — and used by the decode loop to spawn a
/// fresh [`ConstraintState`] per generation via [`ConstraintSpec::build`]. The inner
/// kind is private so the compiled `Schema` stays an implementation detail.
#[derive(Clone, Debug)]
pub struct ConstraintSpec(ConstraintKind);

#[derive(Clone, Debug)]
enum ConstraintKind {
    Json,
    Schema(Schema),
}

impl ConstraintSpec {
    /// The `{"type":"json_object"}` constraint: any valid JSON object.
    pub fn json_object() -> Self {
        Self(ConstraintKind::Json)
    }

    /// Compile a JSON Schema into the supported subset. Returns [`SchemaError`]
    /// (naming the offending keyword) for any feature Camelid cannot enforce
    /// byte-for-byte, so the caller can fail closed with a precise message.
    pub fn from_schema(schema: &serde_json::Value) -> Result<Self, SchemaError> {
        Ok(Self(ConstraintKind::Schema(compile_root(schema)?)))
    }

    /// Spawn a fresh constraint state for one generation.
    pub fn build(&self) -> ConstraintState {
        match &self.0 {
            ConstraintKind::Json => ConstraintState::new_json(),
            ConstraintKind::Schema(schema) => {
                ConstraintState::Schema(SchemaState::new(schema.clone()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed a full string; returns the final state if every byte was accepted.
    fn feed(s: &str) -> Result<JsonState, ()> {
        let mut st = JsonState::new();
        for &b in s.as_bytes() {
            st.advance(b)?;
        }
        Ok(st)
    }

    fn valid_complete(s: &str) -> bool {
        feed(s).map(|st| st.is_done()).unwrap_or(false)
    }

    fn valid_prefix(s: &str) -> bool {
        feed(s).is_ok()
    }

    #[test]
    fn accepts_complete_objects() {
        for s in [
            r#"{}"#,
            r#"{"a":1}"#,
            r#"{ "a" : 1 , "b" : 2 }"#,
            r#"{"a":[1,2,3]}"#,
            r#"{"a":{"b":{"c":[]}}}"#,
            r#"{"s":"hi \"there\"\né"}"#,
            r#"{"n":-12.5e+10,"z":0,"f":false,"t":true,"x":null}"#,
            "{\n  \"k\": 1\n}\n",
        ] {
            assert!(valid_complete(s), "should be complete valid: {s}");
            // serde agrees it is real JSON.
            assert!(serde_json::from_str::<serde_json::Value>(s.trim()).is_ok());
        }
    }

    #[test]
    fn rejects_invalid_bytes() {
        // Each must be rejected at some byte (never a valid prefix all the way).
        for s in [
            "[1]",         // top level must be an object
            "{,}",         // expected key
            "{\"a\":}",    // value required
            "{\"a\":1,}",  // trailing comma -> needs a key
            "{\"a\" 1}",   // missing colon
            "{\"a\":01}",  // leading zero
            "{\"a\":1.}",  // dangling fraction is not complete... but as a prefix it's ok
            "{\"a\":tru}", // bad literal
            "{'a':1}",     // single quotes
        ] {
            // `1.}` is a valid *prefix* until the `}`; assert it is not *complete*.
            assert!(!valid_complete(s), "should not be complete valid JSON: {s}");
        }
        assert!(!valid_prefix("[1]"));
        assert!(!valid_prefix("{,}"));
        assert!(!valid_prefix("{\"a\" 1}"));
        assert!(!valid_prefix("{\"a\":01}"));
        assert!(!valid_prefix("{'a':1}"));
    }

    #[test]
    fn tracks_prefix_validity() {
        // Genuine prefixes of a valid object are accepted but not complete.
        for p in [
            "",
            "{",
            "{\"",
            "{\"a",
            "{\"a\"",
            "{\"a\":",
            "{\"a\":1",
            "{\"a\":1,",
            "{\"a\":-",
            "{\"a\":1.",
            "{\"a\":1e",
            "{\"a\":[",
            "{\"a\":[1,",
            "{\"a\":tru",
        ] {
            assert!(valid_prefix(p), "should be a valid prefix: {p:?}");
            assert!(!valid_complete(p), "prefix should not be complete: {p:?}");
        }
    }

    #[test]
    fn accepts_masks_continuations_without_mutation() {
        let mut st = JsonState::new();
        for &b in br#"{"a":"# {
            st.advance(b).unwrap();
        }
        // A value is required next: these start valid values; `}`/`,` do not.
        assert!(st.accepts(b"1"));
        assert!(st.accepts(b"\"x\""));
        assert!(st.accepts(b"{"));
        assert!(st.accepts(b"true"));
        assert!(!st.accepts(b"}"));
        assert!(!st.accepts(b","));
        // accepts() must not have mutated the live state.
        assert!(!st.is_done());
        st.advance(b'1').unwrap();
        // After the value: ',' or '}' continue/close; another digit does not start.
        assert!(st.accepts(b","));
        assert!(st.accepts(b"}"));
        assert!(!st.accepts(b"x"));
    }

    #[test]
    fn done_only_after_top_level_close() {
        let mut st = JsonState::new();
        for &b in br#"{"a":1}"# {
            st.advance(b).unwrap();
        }
        assert!(st.is_done());
        // Trailing whitespace stays done; anything else is rejected.
        assert!(st.accepts(b"  \n"));
        assert!(!st.accepts(b"{"));
        assert!(!st.accepts(b"1"));
    }

    #[test]
    fn multibyte_tokens_validated_whole() {
        // A token whose bytes span structure ("}," or "1}") must validate as a unit.
        let mut st = JsonState::new();
        for &b in br#"{"a":[1"# {
            st.advance(b).unwrap();
        }
        assert!(st.accepts(b"]}")); // close array then object
        assert!(st.accepts(b",2")); // continue the array
        assert!(!st.accepts(b"}")); // can't close the object while inside the array
    }

    #[test]
    fn constraint_state_json_delegates_to_json_state() {
        // ConstraintState::Json must behave exactly like the underlying JsonState
        // so the Slice 1 seam is a pure pass-through (zero behavior change).
        let mut c = ConstraintState::new_json();
        assert!(!c.is_done());
        // Same masking decisions as a bare JsonState at the top level.
        assert!(c.accepts(b"{"));
        assert!(!c.accepts(b"["));
        for &b in br#"{"a":1}"# {
            c.advance(b).unwrap();
        }
        assert!(c.is_done());
        // Once done, only trailing whitespace is accepted.
        assert!(c.accepts(b"  \n"));
        assert!(!c.accepts(b"{"));

        // Cross-check: the enum agrees byte-for-byte with a directly driven state.
        let mut c2 = ConstraintState::new_json();
        let mut st = JsonState::new();
        for &b in br#"{"k":[1,2]}"# {
            assert_eq!(c2.accepts(&[b]), st.accepts(&[b]));
            c2.advance(b).unwrap();
            st.advance(b).unwrap();
        }
        assert_eq!(c2.is_done(), st.is_done());
        assert!(c2.is_done());
    }

    // ---- JSON Schema subset (SchemaState) ----

    fn schema(v: serde_json::Value) -> Schema {
        compile_root(&v).expect("schema should compile")
    }

    fn feed_schema(s: &Schema, input: &str) -> Result<SchemaState, ()> {
        let mut st = SchemaState::new(s.clone());
        for &b in input.as_bytes() {
            st.advance(b)?;
        }
        Ok(st)
    }

    fn schema_complete(s: &Schema, input: &str) -> bool {
        feed_schema(s, input)
            .map(|st| st.is_done())
            .unwrap_or(false)
    }

    fn schema_prefix(s: &Schema, input: &str) -> bool {
        feed_schema(s, input).is_ok()
    }

    #[test]
    fn compile_rejects_unsupported_shapes() {
        use serde_json::json;
        // Root must be an object or array.
        assert!(compile_root(&json!({"type": "string"})).is_err());
        assert!(compile_root(&json!({"enum": ["a", "b"]})).is_err());
        // Untyped / any property.
        assert!(compile_root(
            &json!({"type": "object", "additionalProperties": false, "properties": {"a": {}}})
        )
        .is_err());
        // Open objects (additionalProperties not false / absent).
        assert!(compile_root(&json!({"type": "object", "properties": {}})).is_err());
        assert!(compile_root(
            &json!({"type": "object", "additionalProperties": true, "properties": {}})
        )
        .is_err());
        // Type unions.
        assert!(compile_root(&json!({"type": ["object", "null"]})).is_err());
        // Combinators / refs.
        assert!(compile_root(&json!({"anyOf": [{"type": "object"}]})).is_err());
        assert!(compile_root(&json!({"$ref": "#/$defs/x"})).is_err());
        // Unenforced constraint keywords must be rejected, not silently ignored.
        assert!(compile_root(&json!({
            "type": "object", "additionalProperties": false,
            "properties": {"a": {"type": "string", "minLength": 1}}
        }))
        .is_err());
        // Non-string enum members.
        assert!(compile_root(&json!({
            "type": "object", "additionalProperties": false,
            "properties": {"a": {"enum": [1, 2]}}, "required": ["a"]
        }))
        .is_err());
        // required referring to an undeclared property.
        assert!(compile_root(&json!({
            "type": "object", "additionalProperties": false,
            "properties": {"a": {"type": "string"}}, "required": ["b"]
        }))
        .is_err());
        // Array without items.
        assert!(compile_root(&json!({"type": "array"})).is_err());
        // Annotations are ignored, not rejected.
        assert!(compile_root(&json!({
            "type": "object", "additionalProperties": false, "title": "T", "description": "d",
            "properties": {"a": {"type": "string", "description": "the a"}}, "required": ["a"]
        }))
        .is_ok());
    }

    #[test]
    fn object_enforces_required_and_value_types() {
        use serde_json::json;
        let s = schema(json!({
            "type": "object", "additionalProperties": false,
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"]
        }));
        assert!(schema_complete(&s, r#"{"name":"bob","age":3}"#));
        assert!(schema_complete(&s, r#"{ "name" : "bob" }"#)); // age optional, whitespace ok

        // Cannot close without the required key.
        assert!(!schema_complete(&s, r#"{}"#));
        assert!(!schema_complete(&s, r#"{"age":3}"#));
        // Wrong value types are rejected at the first offending byte.
        assert!(!schema_prefix(&s, r#"{"name":1"#)); // string expected, got a number
        assert!(!schema_prefix(&s, r#"{"age":""#)); // integer expected, got a string
        assert!(!schema_prefix(&s, r#"{"name":"b","age":1.5"#)); // integer forbids a fraction

        // Unknown key: rejected as soon as it cannot be a declared-property prefix.
        assert!(!schema_prefix(&s, r#"{"x"#));
        // Duplicate key rejected (name already used).
        assert!(!schema_prefix(&s, r#"{"name":"a","n"#));
        // is_done only at the final close.
        assert!(schema_prefix(&s, r#"{"name":"a""#));
        assert!(!schema_complete(&s, r#"{"name":"a""#));
    }

    #[test]
    fn scalars_bool_null_number() {
        use serde_json::json;
        let s = schema(json!({
            "type": "object", "additionalProperties": false,
            "properties": {"b": {"type": "boolean"}, "z": {"type": "null"}, "n": {"type": "number"}},
            "required": ["b", "z", "n"]
        }));
        assert!(schema_complete(&s, r#"{"b":true,"z":null,"n":-12.5e+3}"#));
        assert!(schema_complete(&s, r#"{"b":false,"z":null,"n":0}"#));
        assert!(!schema_prefix(&s, r#"{"b":tru e"#)); // bad literal
        assert!(!schema_prefix(&s, r#"{"b":true,"z":nul,"#)); // bad null
        assert!(!schema_prefix(&s, r#"{"b":true,"z":null,"n":01"#)); // leading zero
    }

    #[test]
    fn string_enum_and_const() {
        use serde_json::json;
        let s = schema(json!({
            "type": "object", "additionalProperties": false,
            "properties": {"unit": {"enum": ["celsius", "fahrenheit"]}},
            "required": ["unit"]
        }));
        assert!(schema_complete(&s, r#"{"unit":"celsius"}"#));
        assert!(schema_complete(&s, r#"{"unit":"fahrenheit"}"#));
        assert!(!schema_prefix(&s, r#"{"unit":"kelvin"#)); // not a candidate
        assert!(!schema_prefix(&s, r#"{"unit":"cel x"#)); // diverges from "celsius"
        let c = schema(json!({
            "type": "object", "additionalProperties": false,
            "properties": {"kind": {"const": "tool"}}, "required": ["kind"]
        }));
        assert!(schema_complete(&c, r#"{"kind":"tool"}"#));
        assert!(!schema_prefix(&c, r#"{"kind":"other"#));
    }

    #[test]
    fn nested_objects_and_arrays() {
        use serde_json::json;
        let s = schema(json!({
            "type": "object", "additionalProperties": false,
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
                "meta": {
                    "type": "object", "additionalProperties": false,
                    "properties": {"n": {"type": "integer"}}, "required": ["n"]
                }
            },
            "required": ["tags", "meta"]
        }));
        assert!(schema_complete(&s, r#"{"tags":["a","b"],"meta":{"n":5}}"#));
        assert!(schema_complete(&s, r#"{"tags":[],"meta":{"n":0}}"#));
        // Array element of the wrong type.
        assert!(!schema_prefix(&s, r#"{"tags":[1"#));
        // Nested required missing.
        assert!(!schema_complete(&s, r#"{"tags":[],"meta":{}}"#));
        // Only complete at the outermost close.
        assert!(schema_prefix(&s, r#"{"tags":[],"meta":{"n":0}"#));
        assert!(!schema_complete(&s, r#"{"tags":[],"meta":{"n":0}"#));
    }

    #[test]
    fn schema_masks_candidates_without_mutation() {
        use serde_json::json;
        let s = schema(json!({
            "type": "object", "additionalProperties": false,
            "properties": {"a": {"type": "integer"}}, "required": ["a"]
        }));
        let mut st = SchemaState::new(s);
        for &b in br#"{"a":"# {
            st.advance(b).unwrap();
        }
        // An integer value must start with '-' or a digit.
        assert!(st.accepts(b"1"));
        assert!(st.accepts(b"-"));
        assert!(!st.accepts(b"\""));
        assert!(!st.accepts(b"{"));
        // accepts() must not mutate the live state.
        assert!(!st.is_done());
        st.advance(b'1').unwrap();
        // After the value, '}' closes (required satisfied); a stray char does not.
        assert!(st.accepts(b"}"));
        assert!(!st.accepts(b"x"));
        assert!(st.accepts(b"23}")); // multibyte token: extend the number then close
    }

    #[test]
    fn object_key_trie_masking() {
        use serde_json::json;
        let s = schema(json!({
            "type": "object", "additionalProperties": false,
            "properties": {"colour": {"type": "string"}, "count": {"type": "integer"}}
        }));
        let mut st = SchemaState::new(s);
        st.advance(b'{').unwrap();
        // A key must start a declared property name.
        assert!(st.accepts(b"\"c"));
        assert!(!st.accepts(b"\"x"));
        // Narrow to "cou": only "count" remains reachable.
        for &b in b"\"cou" {
            st.advance(b).unwrap();
        }
        assert!(st.accepts(b"nt\""));
        assert!(!st.accepts(b"lour\""));
    }

    #[test]
    fn constraint_spec_builds_and_validates() {
        use serde_json::json;
        // A json_object spec builds a working JSON-object constraint.
        let mut c = ConstraintSpec::json_object().build();
        for &b in br#"{"a":1}"# {
            c.advance(b).unwrap();
        }
        assert!(c.is_done());
        // A supported schema compiles and enforces its property types.
        let spec = ConstraintSpec::from_schema(&json!({
            "type": "object", "additionalProperties": false,
            "properties": {"a": {"type": "string"}}, "required": ["a"]
        }))
        .expect("supported schema should compile");
        let s = spec.build();
        assert!(!s.accepts(b"{\"a\":1")); // 'a' must be a string
        assert!(s.accepts(b"{\"a\":\"x\"}"));
        // A schema outside the subset is an error, not a panic or a silent pass.
        assert!(ConstraintSpec::from_schema(&json!({"type": "string"})).is_err());
    }

    #[test]
    fn nullable_type_union_accepts_value_or_null() {
        use serde_json::json;
        // The OpenAI nullable pattern: a required-but-nullable property via ["T","null"].
        let s = schema(json!({
            "type": "object", "additionalProperties": false,
            "properties": {"name": {"type": ["string", "null"]}},
            "required": ["name"]
        }));
        assert!(schema_complete(&s, r#"{"name":"bob"}"#));
        assert!(schema_complete(&s, r#"{"name":null}"#)); // null satisfies the required field

        // Neither a number nor a bad literal is a string-or-null.
        assert!(!schema_prefix(&s, r#"{"name":5"#));
        assert!(!schema_prefix(&s, r#"{"name":t"#));
    }

    #[test]
    fn nullable_integer_and_object_unions() {
        use serde_json::json;
        let s = schema(json!({
            "type": "object", "additionalProperties": false,
            "properties": {
                "age": {"type": ["integer", "null"]},
                "meta": {
                    "type": ["object", "null"], "additionalProperties": false,
                    "properties": {"k": {"type": "string"}}, "required": ["k"]
                }
            },
            "required": ["age", "meta"]
        }));
        assert!(schema_complete(&s, r#"{"age":30,"meta":{"k":"v"}}"#));
        assert!(schema_complete(&s, r#"{"age":null,"meta":null}"#));
        assert!(schema_complete(&s, r#"{"age":5,"meta":null}"#));
        // The nullable object's object branch still enforces its required `k`.
        assert!(!schema_complete(&s, r#"{"age":null,"meta":{}}"#));
        // The integer branch still forbids a fraction.
        assert!(!schema_prefix(&s, r#"{"age":1.5"#));
    }

    #[test]
    fn ambiguous_type_union_is_rejected() {
        use serde_json::json;
        // integer + number share a starting byte, so one lookahead can't disambiguate.
        assert!(compile_root(&json!({
            "type": "object", "additionalProperties": false,
            "properties": {"x": {"type": ["integer", "number"]}}, "required": ["x"]
        }))
        .is_err());
        // A single-element type array is a degenerate (accepted) union.
        assert!(compile_root(&json!({
            "type": "object", "additionalProperties": false,
            "properties": {"x": {"type": ["string"]}}, "required": ["x"]
        }))
        .is_ok());
    }

    #[test]
    fn const_or_enum_with_conflicting_type_is_rejected() {
        use serde_json::json;
        // const/enum members are strings; a contradictory non-string `type` is
        // rejected rather than silently ignored (no dropped constraint).
        assert!(compile_root(&json!({
            "type": "object", "additionalProperties": false,
            "properties": {"k": {"const": "x", "type": "number"}}, "required": ["k"]
        }))
        .is_err());
        assert!(compile_root(&json!({
            "type": "object", "additionalProperties": false,
            "properties": {"k": {"enum": ["a", "b"], "type": "integer"}}, "required": ["k"]
        }))
        .is_err());
        // The review's exact M1 repro, {"type":"integer","enum":["a"]}: pinned
        // explicitly with `type` written first (the fix must not depend on
        // keyword order).
        assert!(compile_root(&json!({
            "type": "object", "additionalProperties": false,
            "properties": {"k": {"type": "integer", "enum": ["a"]}}, "required": ["k"]
        }))
        .is_err());
        // A redundant but consistent `type: "string"` is accepted.
        let s = schema(json!({
            "type": "object", "additionalProperties": false,
            "properties": {"k": {"const": "x", "type": "string"}}, "required": ["k"]
        }));
        assert!(schema_complete(&s, r#"{"k":"x"}"#));
    }

    #[test]
    fn non_ascii_enum_or_const_is_rejected() {
        use serde_json::json;
        // A non-ASCII literal cannot be enforced byte-for-byte: byte-level-BPE and
        // byte-fallback tokenizers emit a multibyte character as byte fragments that
        // decode in isolation to "" and are masked out, so the value could never be
        // produced. It must fail closed at compile time (a request-time 400) rather than
        // dead-ending mid-generation. `\u{e9}` = 'é', `\u{2103}` = '℃'.
        assert!(compile_root(&json!({
            "type": "object", "additionalProperties": false,
            "properties": {"city": {"enum": ["caf\u{e9}", "berlin"]}}, "required": ["city"]
        }))
        .is_err());
        assert!(compile_root(&json!({
            "type": "object", "additionalProperties": false,
            "properties": {"unit": {"const": "\u{2103}"}}, "required": ["unit"]
        }))
        .is_err());
        // A pure-ASCII enum/const is unaffected.
        let s = schema(json!({
            "type": "object", "additionalProperties": false,
            "properties": {"unit": {"enum": ["celsius", "fahrenheit"]}}, "required": ["unit"]
        }));
        assert!(schema_complete(&s, r#"{"unit":"celsius"}"#));
    }

    /// An object schema with `n` integer properties named `p0000`..
    fn object_with_n_properties(n: usize) -> serde_json::Value {
        let mut props = serde_json::Map::new();
        for i in 0..n {
            props.insert(format!("p{i:04}"), serde_json::json!({"type": "integer"}));
        }
        serde_json::json!({
            "type": "object", "additionalProperties": false, "properties": props
        })
    }

    /// An object schema whose single property is an enum with `n` members.
    fn object_with_enum_members(n: usize) -> serde_json::Value {
        let members: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
        serde_json::json!({
            "type": "object", "additionalProperties": false,
            "properties": {"e": {"enum": members}}, "required": ["e"]
        })
    }

    /// A schema nested `levels` objects deep (each level one property `a`).
    /// The innermost node is an integer, so the deepest node sits at depth
    /// `levels` (the root object is depth 0).
    fn nested_object_schema(levels: usize) -> serde_json::Value {
        let mut node = serde_json::json!({"type": "integer"});
        for _ in 0..levels {
            node = serde_json::json!({
                "type": "object", "additionalProperties": false,
                "properties": {"a": node}, "required": ["a"]
            });
        }
        node
    }

    #[test]
    fn schema_dimension_caps_reject_oversized_and_accept_at_cap() {
        use serde_json::json;
        // Property count: 256 per object compiles, 257 is a typed error.
        assert!(compile_root(&object_with_n_properties(MAX_OBJECT_PROPERTIES)).is_ok());
        let err = compile_root(&object_with_n_properties(MAX_OBJECT_PROPERTIES + 1))
            .expect_err("over-cap property count must be rejected");
        assert!(err.to_string().contains("properties"), "{err}");

        // Property-name length: 64 bytes compiles, 65 is rejected.
        let ok_name = "n".repeat(MAX_PROPERTY_NAME_BYTES);
        assert!(compile_root(&json!({
            "type": "object", "additionalProperties": false,
            "properties": {ok_name: {"type": "integer"}}
        }))
        .is_ok());
        let long_name = "n".repeat(MAX_PROPERTY_NAME_BYTES + 1);
        let err = compile_root(&json!({
            "type": "object", "additionalProperties": false,
            "properties": {long_name: {"type": "integer"}}
        }))
        .expect_err("over-cap property name must be rejected");
        assert!(err.to_string().contains("property name"), "{err}");

        // Enum member count: 256 compiles, 257 is rejected.
        assert!(compile_root(&object_with_enum_members(MAX_ENUM_MEMBERS)).is_ok());
        let err = compile_root(&object_with_enum_members(MAX_ENUM_MEMBERS + 1))
            .expect_err("over-cap enum member count must be rejected");
        assert!(err.to_string().contains("enum"), "{err}");

        // Enum member size (JSON-encoded, quotes included): 254 chars encode to
        // exactly 256 bytes and compile; one more byte is rejected. `const` shares
        // the cap via the same literal compiler.
        let at_cap = "x".repeat(MAX_ENUM_MEMBER_BYTES - 2);
        assert!(compile_root(&json!({
            "type": "object", "additionalProperties": false,
            "properties": {"e": {"enum": [at_cap]}}
        }))
        .is_ok());
        let over_cap = "x".repeat(MAX_ENUM_MEMBER_BYTES - 1);
        let err = compile_root(&json!({
            "type": "object", "additionalProperties": false,
            "properties": {"e": {"const": over_cap}}
        }))
        .expect_err("over-cap literal must be rejected");
        assert!(err.to_string().contains("bytes"), "{err}");

        // Nesting depth: a leaf at depth 32 compiles, depth 33 is rejected.
        assert!(compile_root(&nested_object_schema(MAX_SCHEMA_DEPTH)).is_ok());
        let err = compile_root(&nested_object_schema(MAX_SCHEMA_DEPTH + 1))
            .expect_err("over-cap nesting depth must be rejected");
        assert!(err.to_string().contains("depth"), "{err}");
    }

    #[test]
    fn exhausted_object_rejects_comma_and_key_quote() {
        use serde_json::json;
        // Review repro (B2): once every declared property is used, `,` (and the
        // common BPE token `,"`) must be masked out — otherwise the next step's
        // mask is all-false and an ordinary schema 422s mid-generation.
        let s = schema(json!({
            "type": "object", "additionalProperties": false,
            "properties": {"a": {"type": "integer"}}, "required": ["a"]
        }));
        let st = feed_schema(&s, r#"{"a":1"#).expect("valid prefix");
        assert!(!st.accepts(b","));
        assert!(!st.accepts(b",\"")); // the two-byte BPE token from the review
        assert!(st.accepts(b"}"));
    }

    #[test]
    fn empty_object_schema_rejects_key_quote() {
        use serde_json::json;
        // Sibling dead-end: a zero-property object must reject `"` right after `{`
        // (no property could ever complete the key) but still close with `}`.
        let s = schema(json!({
            "type": "object", "additionalProperties": false, "properties": {}
        }));
        let st = feed_schema(&s, "{").expect("valid prefix");
        assert!(!st.accepts(b"\""));
        assert!(st.accepts(b"}"));
        assert!(schema_complete(&s, "{}"));
    }

    #[test]
    fn partial_object_still_accepts_comma() {
        use serde_json::json;
        // The unused-property gate must not be over-broad: with a declared
        // property still unused, `,` (and the next key) stays legal.
        let s = schema(json!({
            "type": "object", "additionalProperties": false,
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"]
        }));
        let st = feed_schema(&s, r#"{"a":1"#).expect("valid prefix");
        assert!(st.accepts(b","));
        assert!(st.accepts(b",\"b\":2}"));
        assert!(schema_complete(&s, r#"{"a":1,"b":2}"#));
    }

    #[test]
    fn advance_error_is_sticky_failed() {
        use serde_json::json;
        // A rejected byte must poison the state, not park it in Done: `advance`
        // dispatches via mem::replace(mode, Done), so without the Failed sink a
        // truncated/invalid value would report is_done() and finish as "stop".
        let s = schema(json!({
            "type": "object", "additionalProperties": false,
            "properties": {"a": {"type": "integer"}}, "required": ["a"]
        }));
        let mut st = SchemaState::new(s);
        assert!(st.advance(b'x').is_err()); // '{' required
        assert!(!st.is_done());
        // Every subsequent byte — including otherwise-legal ones — is rejected.
        assert!(!st.accepts(b"{"));
        assert!(!st.accepts(b" "));
        assert!(st.advance(b'{').is_err());
        assert!(!st.is_done());
    }
}
