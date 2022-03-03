CREATE DATABASE makers;

CREATE TABLE makers.trades
(
    `id` UInt64,
    `timestamp` DateTime64(3),
    `exchange` String,
    `symbol` String,
    `side` String,
    `amount` Float64,
    `price` Float64
)
ENGINE = MergeTree
PRIMARY KEY (id, exchange)
ORDER BY (id, exchange)
SETTINGS index_granularity = 8192;

CREATE TABLE makers.books
(
    `timestamp` DateTime64(3),
    `exchange` String,
    `symbol` String,
    `bid` Map(String, LowCardinality(Float64)),
    `ask` Map(String, LowCardinality(Float64)),
    INDEX exchange_index lowerUTF8(exchange) TYPE tokenbf_v1(65536, 3, 0) GRANULARITY 8192
)
ENGINE = MergeTree
PRIMARY KEY (timestamp, exchange)
ORDER BY (timestamp, exchange)
SETTINGS index_granularity = 8192;

CREATE TABLE makers.trades_queue (
        exchange String,
        symbol String,
        timestamp Float64,
        side String,
        amount Float64,
        price Float64,
        order_type Nullable(String),
        id UInt64
        ) ENGINE = Kafka
SETTINGS kafka_broker_list = 'gruenau2:9092',
       kafka_topic_list = 'trades-BITFINEX-BTC-USD,trades-BITFINEX-ETH-USD,trades-COINBASE-BTC-USD,trades-COINBASE-ETH-USD,trades-GEMINI-BTC-USD,trades-GEMINI-ETH-USD,trades-POLONIEX-BTC-USDT,trades-POLONIEX-ETH-USDT',
       kafka_group_name = 'test_group',
       kafka_format = 'JSONEachRow',
       kafka_row_delimiter = '\n';


CREATE TABLE makers.books_queue (
        exchange String,
        symbol String,
        timestamp Float64,
        book Map(String, Map(String, LowCardinality(Float64)))
        ) ENGINE = Kafka
SETTINGS kafka_broker_list = 'gruenau2:9092',
       kafka_topic_list = 'book-BITFINEX-BTC-USD,book-BITFINEX-ETH-USD,book-COINBASE-BTC-USD,book-COINBASE-ETH-USD,book-GEMINI-BTC-USD,book-GEMINI-ETH-USD,book-POLONIEX-BTC-USDT,book-POLONIEX-ETH-USDT',
       kafka_group_name = 'test_group',
       kafka_format = 'JSONEachRow',
       kafka_row_delimiter = '\n';


CREATE MATERIALIZED VIEW makers.books_queue_mv TO makers.books
(
    `timestamp` Float64,
    `exchange` String,
    `symbol` String,
    `bid` Map(String, LowCardinality(Float64)),
    `ask` Map(String, LowCardinality(Float64))
) AS
SELECT
    timestamp,
    exchange,
    symbol,
    CAST(book['bid'], 'Map(String, LowCardinality(Float64))') AS bid,
    CAST(book['ask'], 'Map(String, LowCardinality(Float64))') AS ask
FROM makers.books_queue;


CREATE MATERIALIZED VIEW makers.trades_queue_mv TO makers.trades
(
    `id` UInt64,
    `timestamp` Float64,
    `exchange` String,
    `symbol` String,
    `side` String,
    `amount` Float64,
    `price` Float64
) AS
SELECT
    id,
    timestamp,
    exchange,
    symbol,
    side,
    amount,
    price
FROM makers.trades_queue;
