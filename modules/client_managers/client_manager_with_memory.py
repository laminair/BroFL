from itertools import groupby
from operator import itemgetter

from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import SimpleClientManager


class MemoryClientManager(SimpleClientManager):
    
    def __init__(self) -> None:
        super().__init__()
        self.client_memory = {}
        self.unused_indices = []

    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy

        Returns
        -------
        success : bool
            Indicating if registration was successful. False if ClientProxy is
            already registered or can not be registered for any reason.
        """
        if client.cid in self.clients:
            return False

        # Here, we introduce a client enumeration to control data shard selection on clients.
        print("Clients registered: ", len(self.client_memory) + 1)
        if client.cid not in self.client_memory.keys():
            self.client_memory[client.cid] = {
                "client_id": self.assign_client_id(self.client_memory)
            }

        self.clients[client.cid] = client
        with self._cv:
            self._cv.notify_all()

        return True

    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy
        """
        if client.cid in self.clients:
            del self.clients[client.cid]
            # Here we remove the client index and allow another client to take the index
            del self.client_memory[client.cid]

            with self._cv:
                self._cv.notify_all()

    def check_clients_are_enumerated_consecutively(self, client_ids: list):
        """
        With this function we check if all client IDs are in consecutive order. If not, we return a list of unused
        indices that can be used to fill spots until we get a consecutive ordering again. This is primarily for
        experiment reasons as we run distributed data shards controlled by client IDs.
        :param client_ids: A list of client_id's (list)
        """
        list_of_lists = []

        if len(self.unused_indices) > 0:
            return False, self.unused_indices

        for key, group in groupby(enumerate(client_ids), lambda ix: ix[0] - ix[1]):
            client_id_list = list(map(itemgetter(1), group))
            list_of_lists.append(client_id_list)

        if len(list_of_lists) > 1:
            list_dict = {}
            unused_indices = []
            for idx, sublist in enumerate(list_of_lists):
                list_dict[idx] = {"min": min(sublist), "max": max(sublist)}

                if idx > 0:
                    ctr = list_dict[idx - 1]["max"]
                    while ctr < list_dict[idx]["min"] - 1:
                        ctr += 1
                        unused_indices.append(ctr)

            return False, sorted(unused_indices)

        else:
            return True, []

    def assign_client_id(self, client_memory):
        """
        Here we return the correct client ID based on the state of the client_memory, i.e. what clients are currently
        registered with the server.
        """
        indices = [val["client_id"] for val in client_memory.values()]
        clients_enumerated_consecutively, unused_indices = self.check_clients_are_enumerated_consecutively(indices)

        if clients_enumerated_consecutively:
            return len(client_memory)
        else:
            # Here we use client enumerations to restore consecutive numbering should a client drop out of the training.
            # Clients are considered ephemeral. In our experiments, they generally have the same compute performance.
            try:
                c_id = unused_indices.pop(0)
            except IndexError:
                c_id = len(client_memory)

            return c_id
