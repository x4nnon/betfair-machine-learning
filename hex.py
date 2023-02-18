from utils.config import BATH_HEX
from paramiko import (
    SSHClient,
    AutoAddPolicy,
    AuthenticationException,
    SSHException,
)


def connect_ssh_client(
    hostname: str, username: str, password: str
) -> (SSHClient | None):
    # Set up SSH client
    client = SSHClient()
    client.set_missing_host_key_policy(AutoAddPolicy())

    # Connect to SSH server
    # Connect to SSH server and check for errors
    try:
        client.connect(hostname=hostname, username=username, password=password)
        print("Login successful!")
        return client
    except AuthenticationException:
        print("Authentication failed, please verify your credentials")
        client.close()
        exit()
    except SSHException as sshException:
        print("Unable to establish SSH connection: %s" % sshException)
        client.close()
        exit()
    except Exception as e:
        print("Error occurred while connecting to %s: %s" % (hostname, e))
        client.close()
        exit()


def execute_commands(commands: list[str], client: SSHClient):
    for command in commands:
        stdin, stdout, stderr = client.exec_command(command)
        try:
            # print the output of the command
            print(stdout.read().decode())
        except Exception as e:
            print(f"Error executing command: {command}")
            print(f"Error message: {str(e)}")
        finally:
            if stdin:
                stdin.close()
            if stdout:
                stdout.close()
            if stderr:
                stderr.close()


if __name__ == "__main__":
    # If no exceptions are raised, the login was successful

    username, password, hostname, docker_image = BATH_HEX.values()
    container_name = ""
    image_name = "bet-ml"

    client = connect_ssh_client(hostname, username, password)

    # # Run the commands to build the Docker image and save it
    commands = [
        f"hare build -t {username}/{image_name} ."  # make sure dockerfile is in right place
        # f'hare run -it --gpus device=0  -v "$(pwd)":/app {container_name} bash',
        # "exit",
        "hare me",
        # f"hare commit {container_name} {username}/{image_name}",
        # f"hare rm {username}/{image_name}",
    ]

    execute_commands(commands, client)

    # close the SSH connection
    if client:
        client.close()
