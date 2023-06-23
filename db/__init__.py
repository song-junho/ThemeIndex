from direct_redis import DirectRedis

# redis 생성
redis_client = DirectRedis(host="127.0.0.1", port=6379, db=0, max_connections=4)
