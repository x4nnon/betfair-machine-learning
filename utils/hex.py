import yaml
from config import BATH_HEX
from paramiko import (
    SSHClient,
    AutoAddPolicy,
    AuthenticationException,
    SSHException,
)


class Hex:
    def __init__(self, hostname, username, password):
        self.hostname = hostname
        self.username = username
        self.__password = password
        self.client = self.__connect_ssh_client(hostname, username, password)

    def __connect_ssh_client(
        self, hostname: str, username: str, password: str
    ):  # -> (SSHClient | None):
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

    def __execute_command(self, command: str):
        stdin, stdout, stderr = self.client.exec_command(command)
        try:
            output = stdout.read().decode()
            return yaml.safe_load(output)
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

    def delete_containers(self, docker_image_name):
        out = self.__execute_command("hare me")
        containers = out["containers"].split("|")

        # containers = containers.split("|")[0].strip()
        containers = [
            container
            for container in containers
            if f"{self.username}/{docker_image_name}" != container
            and "bash" not in container
        ]
        for container in containers:
            self.__execute_command(f"hare rm {container}")

    def delete_image(self, docker_image_name):
        out = self.__execute_command(f"hare rmi {self.username}/{docker_image_name}")
        print(out)

    def build_image(self, docker_image_name, directory="."):
        out = self.__execute_command(
            f"hare build -t {self.username}/{docker_image_name} {directory}"
        )
        print(out)

    def run_container(self, username, docker_image_name, device):
        out = self.__execute_command(
            f'hare run -it --gpus device={device}  -v "$(pwd)":/app {username}/{docker_image_name} bash'
        )
        print(out)

    def exit_container(self):
        self.__execute_command("exit")

    def close_client(self):
        self.client.close()

    def run_bet_ml_script(self, docker_image_name: str, device=0):
        self.__execute_command("cd Bet-ML")
        self.build_image(self.username, docker_image_name)
        self.run_container(self.username, docker_image_name, device)
        # self.__execute_command(f"python {file_name}")
        # self.exit_container()
        self.delete_container()
        self.delete_image(docker_image_name)
        self.close_client()


if __name__ == "__main__":
    # If no exceptions are raised, the login was successful

    username, password, hostname, docker_image_name = BATH_HEX.values()
    container_name = ""
    b_hex = Hex(hostname=hostname, username=username, password=password)

    b_hex.run_bet_ml_script(docker_image_name)
