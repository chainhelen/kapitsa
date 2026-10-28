use kapitsa_sdk::Engine;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: chat /local/model-directory")?;
    // Escape the path without introducing a JSON dependency into the SDK.
    let escaped: String = path
        .chars()
        .flat_map(|c| match c {
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\\' => "\\\\".chars().collect(),
            c if c.is_control() => format!("\\u{:04x}", c as u32).chars().collect(),
            c => vec![c],
        })
        .collect();
    let engine = Engine::new(&format!(r#"{{"model_dir":"{escaped}","slots":1}}"#))?;
    for prompt in ["你好", "用一句话解释 Attention"] {
        let request = engine.request(&format!(
            r#"{{"messages":[{{"role":"user","content":"{prompt}"}}],"max_tokens":64}}"#
        ))?;
        let output = request.generate(|delta| {
            println!("{delta}");
            true
        })?;
        println!("{}", output.json);
    }
    Ok(())
}
