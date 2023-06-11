import psycopg2
import socket


class PowerDBHandler(object):
    """
    This class is used to query our internal Power Monitoring Database for our embedded devices.
    """

    def __init__(self):
        # The database is running on PSQL 12.
        # It is only available from inside our chair networks.
        self.connection_string = f"host=collector.dis.cit.tum.de port=5432 user=msrg_ro password=edge_computing_2022 dbname=edge_computing_monitoring"
        self.db_table = "switch_stats"
        self.db_cursor = None
        self.db_connection = None

    def read_latest_power_metric(self):
        if self.db_cursor is None:
            self.connect_to_db()

        if self.db_cursor is None:
            return None

        hostname = socket.gethostname()
        # if "raspi" not in hostname or "jnano" not in hostname:
        #     # As we can only measure the embedded devices' power consumption via the Monitoring DB at this stage, we
        #     # only return values for the Raspis and the Jetson Nanos.
        #     return None

        fqdn = f"{hostname}.exp.dis.cit.tum.de"
        query = "SELECT power_draw_mw " \
                "FROM switch_stats " \
                "WHERE hostname = %s AND " \
                "ts = (SELECT MAX(ts) FROM switch_stats WHERE hostname = %s)"

        self.db_cursor.execute(query, (fqdn, fqdn))
        res = self.db_cursor.fetchone()

        if type(res) is tuple:
            res = res[0]
        else:
            res = 0

        return res

    def connect_to_db(self):
        """
        Connects to the DB. Needs to be called for every query you want to run. Make sure to properly open and close the
        connection. With this, we get better control of network traffic.
        """
        try:
            self.db_connection = psycopg2.connect(self.connection_string)
            self.db_cursor = self.db_connection.cursor()

        except psycopg2.OperationalError as e:
            print(e)
            return None

    def close_db_connection(self):
        # We do not need to commit here as we only read from the DB.
        self.db_connection.close()
        self.db_cursor.close()


if __name__ == "__main__":
    db_handler = PowerDBHandler()
    # db_handler.connect_to_db()
    metric = db_handler.read_latest_power_metric()
    print(metric)

