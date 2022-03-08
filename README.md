# Data collection

On the remote machine:
  
    # start zookeeper
    `bin/zookeeper-server-start.sh config/zookeeper.properties`
    
    # start kafka
    bin/kafka-server-start.sh config/server.properties
    
    # run python script
    python main.py
    
One the local machine:

    # open ssh tunnel
    ssh -N [user@host] -L 9092:localhost:9092
    
    # start clickhouse container
    docker-compose up

# Data snippet

Links to 12h of trading data (Put it into `_input/data/` directory).

Book data:
https://mega.nz/file/JYMQDR5S#NPrcm6S9Okl0O6jdeZr1A0WlABvGd03ejjWCCScLJDU

Trade data:
https://mega.nz/file/JZUCxTyL#ec_lNfXGYEBhWg3BEdjiaZdhMRDFjlhTN_9-zZ_U5sQ
