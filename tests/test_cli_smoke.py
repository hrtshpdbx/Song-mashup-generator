from mashup.cli import build_parser


def test_cli_parser():
    parser = build_parser()

    args = parser.parse_args(
        [
            "https://youtu.be/a",
            "https://youtu.be/b",
            "--output-dir",
            "outputs/test",
            "--output-name",
            "test.wav",
        ]
    )

    assert args.url1 == "https://youtu.be/dQw4w9WgXcQ?feature=shared"
    assert args.url2 == "https://youtu.be/KvknOXGPzCQ?feature=shared"
    assert args.output_dir == "outputs/test"
    assert args.output_name == "test.wav"
