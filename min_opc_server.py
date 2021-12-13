import sys
import time
from opcua import ua, Server
import argparse
sys.path.insert(0, "..")


def get_args():

    parser = argparse.ArgumentParser(
        description="ip and port selection...")
    parser.add_argument(
        '-ip', '--ip',
        help="indicate ip address", default='localhost', type=str)
    parser.add_argument(
        '-p', '--port',
        help="indicate port", default='4840', type=str)
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":

    kwargs = get_args()

    # setup our server
    server = Server()
    server.set_endpoint(f"opc.tcp://{kwargs['ip']}:{kwargs['port']}/freeopcua/server/")

    # setup our own namespace, not really necessary but should as spec
    uri = "http://examples.freeopcua.github.io"
    idx = server.register_namespace(uri)

    # get Objects node, this is where we should put our nodes
    objects = server.get_objects_node()

    # populating our address space
    myobj = objects.add_object(idx, "MyObject")
    myvar = myobj.add_variable(idx, "MyVariable", 6.7)
    myvar.set_writable()    # Set MyVariable to be writable by clients

    # starting!
    server.start()

    try:
        count = 0
        while True:
            time.sleep(1)
            count += 0.1
            myvar.set_value(count)
    finally:
        #close connection, remove subcsriptions, etc
        server.stop()