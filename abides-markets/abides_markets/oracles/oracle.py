from abides_core import NanosecondTime


class Oracle:
    def get_daily_open_price(
        self, symbol: str, mkt_open: NanosecondTime, cents: bool = True
    ) -> int:
        raise NotImplementedError
