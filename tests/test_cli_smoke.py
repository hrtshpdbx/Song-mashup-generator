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

    assert args.song1_url == "https://youtu.be/a"
    assert args.song2_url == "https://youtu.be/b"
    assert args.output_dir == "outputs/test"
    assert args.output_name == "test.wav"


def test_cli_parser_accepts_local_files():
    args = build_parser().parse_args(["/data/a.webm", "/data/b.webm", "--output-dir", "out",
                                      "--human-mashup-url", "/data/m.webm"])
    assert args.song1_url == "/data/a.webm"
    assert args.human_mashup_url == "/data/m.webm"
