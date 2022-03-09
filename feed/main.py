'''
Copyright (C) 2018-2022 Bryant Moscon - bmoscon@gmail.com
Please see the LICENSE file for the terms and conditions
associated with this software.
'''
from cryptofeed import FeedHandler
from cryptofeed.backends.kafka import BookKafka, TradeKafka
from cryptofeed.defines import L2_BOOK, TRADES
from cryptofeed.exchanges import Coinbase, Bitfinex, Binance, Poloniex, Gemini, Deribit, FTX, Kraken


"""
You can run a consumer in the console with the following command
(assuminng the defaults for the consumer group and bootstrap server)
$ kafka-console-consumer --bootstrap-server 127.0.0.1:9092 --topic trades-COINBASE-BTC-USD
"""

class CustomBookKafka(BookKafka):
    # interval in seconds
    def __init__(self, *args, interval, **kwargs):
        self.last_write = 0
        self.interval = interval
        super().__init__(*args, **kwargs)
    
    async def __call__(self, book, receipt_timestamp: float):
        timestamp = book.timestamp or receipt_timestamp
        if self.snapshots_only and self.last_write + self.interval < timestamp:
            self.last_write = timestamp
            await self._write_snapshot(book, receipt_timestamp)


def main():
    
    f = FeedHandler()
    #cbs = {TRADES: TradeKafka(), L2_BOOK: CustomBookKafka(snapshots_only=True, interval=0.005)}
    cbs = {TRADES: TradeKafka(), L2_BOOK: BookKafka(snapshots_only=True)}

    f.add_feed(Coinbase(max_depth=10, channels=[TRADES, L2_BOOK], symbols=['BTC-USD', 'ETH-USD', 'ETH-BTC'], callbacks=cbs))
    f.add_feed(Bitfinex(max_depth=10, channels=[TRADES, L2_BOOK], symbols=['BTC-USD', 'ETH-USD', 'ETH-BTC'], callbacks=cbs))
    f.add_feed(Poloniex(max_depth=10, channels=[TRADES, L2_BOOK], symbols=['BTC-USDT', 'ETH-USDT', 'ETH-BTC'], callbacks=cbs))
    f.add_feed(Gemini(max_depth=10, channels=[TRADES, L2_BOOK], symbols=['BTC-USD', 'ETH-USD', 'ETH-BTC'], callbacks=cbs))
    #f.add_feed(Deribit(max_depth=10, channels=[TRADES, L2_BOOK], symbols=['BTC-USD', 'ETH-USD', 'ETH-BTC'], callbacks=cbs))
    f.add_feed(FTX(max_depth=10, channels=[TRADES, L2_BOOK], symbols=['BTC-USD', 'ETH-USD', 'ETH-BTC'], callbacks=cbs))
    f.add_feed(Kraken(max_depth=10, channels=[TRADES, L2_BOOK], symbols=['BTC-USD', 'ETH-USD', 'ETH-BTC'], callbacks=cbs))
    f.run()


if __name__ == '__main__':
    main()
