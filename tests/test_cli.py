from aerial_geoloc.cli import build_parser


def test_parser_train_defaults():
    parser = build_parser()
    args = parser.parse_args(["train", "--config", "configs/demo.yaml"])
    assert args.command == "train"
    assert args.weather_aug is None
    assert str(args.config).endswith("demo.yaml")


def test_parser_weather_flags():
    parser = build_parser()
    on = parser.parse_args(["train", "--weather-aug"])
    off = parser.parse_args(["train", "--no-weather-aug"])
    assert on.weather_aug is True
    assert off.weather_aug is False


def test_parser_study_and_sim():
    parser = build_parser()
    study = parser.parse_args(["study"])
    assert study.command == "study"
    assert str(study.config).endswith("study.yaml")
    sim = parser.parse_args(["sim-generate", "--root", "data/sim"])
    assert sim.command == "sim-generate"


def test_parser_retrieve():
    parser = build_parser()
    args = parser.parse_args(["retrieve", "--topk", "3", "--max-queries", "2"])
    assert args.topk == 3
    assert args.max_queries == 2
