from reglens.config import Settings


def test_dotenv_preserves_prefix_spaces(tmp_path):
    env = tmp_path / ".env"
    env.write_text('QUERY_PREFIX="query: "\nLLM_PROVIDERS=groq, gemini\nTOP_K=7\n', encoding="utf-8")
    s = Settings(_env_file=str(env))
    assert s.query_prefix == "query: " and s.provider_names == ["groq", "gemini"] and s.top_k == 7
