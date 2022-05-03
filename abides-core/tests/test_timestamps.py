from abides_core.utils import str_to_ns


def test_str_to_ns():
    assert str_to_ns("0") == 0
    assert str_to_ns("1") == 1

    assert str_to_ns("1us") == 1e3
    assert str_to_ns("1ms") == 1e6

    assert str_to_ns("1s") == 1e9
    assert str_to_ns("1sec") == 1e9
    assert str_to_ns("1second") == 1e9

    assert str_to_ns("1m") == 1e9 * 60
    assert str_to_ns("1min") == 1e9 * 60
    assert str_to_ns("1minute") == 1e9 * 60

    assert str_to_ns("1h") == 1e9 * 60 * 60
    assert str_to_ns("1hr") == 1e9 * 60 * 60
    assert str_to_ns("1hour") == 1e9 * 60 * 60

    assert str_to_ns("1d") == 1e9 * 60 * 60 * 24
    assert str_to_ns("1day") == 1e9 * 60 * 60 * 24

    assert str_to_ns("00:00:00") == 0
    assert str_to_ns("00:00:01") == 1e9
    assert str_to_ns("00:01:00") == 1e9 * 60
    assert str_to_ns("01:00:00") == 1e9 * 60 * 60
