from mongoengine import connect, disconnect

class DatabaseHandler():

    @classmethod
    def connect_to_atlas(cls, username, password, cluster_url, retryWrites=True, w="majority"):
        connect(host=f"mongodb+srv://{username}:{password}@{cluster_url}/?retryWrites={str(retryWrites).lower()}&w={w}")

    @classmethod
    def connect_to_local(cls, collection_name, retryWrites=True, w="majority"):
        connect(collection_name)

    @classmethod
    def connect_over_network(cls, username, password, ip, collection_name, port=27017, retryWrites=True, w="majority"):
        connect(host=f"mongodb://{username}:{password}@{ip}:{port}/{collection_name}?retryWrites={str(retryWrites).lower()}&w={w}")

    @classmethod
    def disconnect(cls):
        disconnect()
