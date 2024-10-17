import os

import duckdb
import fire
from index_generator.index_generator import IndexGenerator
from query.query import Query
from registration.registration import Registration
from summarizer.summarizer import Summarizer
from utils.storage_config import get_storage_path


class Pneuma:
    def __init__(
        self,
        out_path: str = get_storage_path(),
        hf_token: str = "",
    ):
        os.makedirs(out_path, exist_ok=True)
        self.out_path = out_path
        self.db_path = out_path + "/storage.db"
        self.index_location = out_path + "/indexes"
        self.hf_token = hf_token

        self.registration = None
        self.summarizer = None
        self.index_generator = None
        self.query = None

    def __init_registration(self):
        self.registration = Registration(db_path=self.db_path)

    def __init_summarizer(self):
        self.summarizer = Summarizer(db_path=self.db_path, hf_token=self.hf_token)

    def __init_index_generator(self):
        self.index_generator = IndexGenerator(
            db_path=self.db_path, index_path=self.index_location
        )

    def __init_query(self):
        self.query = Query(
            db_path=self.db_path, index_path=self.index_location, hf_token=self.hf_token
        )

    def setup(self) -> str:
        if self.registration is None:
            self.__init_registration()
        return self.registration.setup()

    def add_tables(
        self,
        path: str,
        creator: str,
        source: str = "file",
        s3_region: str = None,
        s3_access_key: str = None,
        s3_secret_access_key: str = None,
        accept_duplicates: bool = False,
    ) -> str:
        if self.registration is None:
            self.__init_registration()
        return self.registration.add_tables(
            path,
            creator,
            source,
            s3_region,
            s3_access_key,
            s3_secret_access_key,
            accept_duplicates,
        )

    def add_metadata(
        self,
        metadata_path: str = "",
        metadata_type: str = "",
        table_id: str = "",
    ) -> str:
        if self.registration is None:
            self.__init_registration()
        return self.registration.add_metadata(metadata_path, metadata_type, table_id)

    def summarize(self, table_id: str = "") -> str:
        if self.summarizer is None:
            self.__init_summarizer()
        return self.summarizer.summarize(table_id)

    def purge_tables(self) -> str:
        if self.summarizer is None:
            self.__init_summarizer()
        return self.summarizer.purge_tables()

    def generate_index(self, index_name: str, table_ids: list | tuple = None) -> str:
        if self.index_generator is None:
            self.__init_index_generator()
        return self.index_generator.generate_index(index_name, table_ids)

    def query_index(self, index_name: str, query: str, k: int = 10) -> str:
        if self.query is None:
            self.__init_query()
        return self.query.query(index_name, query, k)

    def sanity_check(self) -> str:
        return "This works!"


def main():
    print("Hello From Pneuma Main")
    fire.Fire(Pneuma)


if __name__ == "__main__":
    main()
