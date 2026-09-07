from aerial_geoloc.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["eval", *(__import__("sys").argv[1:])]))
