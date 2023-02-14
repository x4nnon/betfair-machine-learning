from io import StringIO
import json
import sys
import pandas as pd
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.file import File
from utils.config import app_principal, SITE_URL
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.constants import ANALYSIS_FILES_TEST, ANALYSIS_FILES_TRAIN


class Onedrive:
    def __init__(
        self,
        client_id,
        client_secret,
        site_url,
    ):
        self.site_url = site_url
        self.auth = self.__authenticate(client_id, client_secret)

    def __authenticate(
        self, client_id: str, client_secret: str
    ) -> list[ClientContext.web, ClientContext]:

        ctx_auth = AuthenticationContext(self.site_url)
        if ctx_auth.acquire_token_for_app(
            client_id=client_id,
            client_secret=client_secret,
        ):
            ctx = ClientContext(self.site_url, ctx_auth)
            web = ctx.web
            ctx.load(web)
            ctx.execute_query()
            print("Authenticated into sharepoint app for: ", web.properties["Title"])
            return ctx, web

        else:
            print(ctx_auth.get_last_error())
            sys.exit()

    def __load_execute(self, content):
        ctx, _ = self.auth
        ctx.load(content)
        ctx.execute_query()

    def __get_file_data(self, f, ctx, csv):

        file_relative_url = f.properties["ServerRelativeUrl"]

        if not csv:
            f_name = f.properties["Name"]
            binary_content = f.open_binary(ctx, file_relative_url).content
            return binary_content

            # with open(f"test_folder/{f_name}", "wb") as local_file:
            #     local_file.write(binary.content)
        else:
            binary_content = f.open_binary(ctx, file_relative_url).content
            data = (
                pd.read_csv(StringIO(binary_content.decode()))
                if csv
                else binary_content
            )
        return data

    def get_folder_contents(
        self,
        target_folder: str,
        target_file=None,
        only_folder_names=False,
        only_folder_files=False,
        csv=True,
    ) -> pd.DataFrame:
        """Returns contents of a target folder in a pandas DataFrame

        Args:
            target_folder (str): name of the folder

        Returns:
            pd.DataFrame: folder content
        """
        try:
            df = pd.DataFrame()
            ctx, web = self.auth
            list_object = web.lists.get_by_title("Documents")
            folder = list_object.root_folder
            self.__load_execute(folder)

            folders = folder.folders
            self.__load_execute(folders)

            target_folder_files = []
            for folder in folders:
                folder_name = folder.properties["Name"]
                if target_folder == folder_name:
                    files = folder.files
                    self.__load_execute(files)

                    if only_folder_names:
                        f_names = []
                        for f in files:
                            f_name = f.properties["Name"]
                            f_names.append(f_name)

                        return f_names

                    if only_folder_files:
                        with ThreadPoolExecutor() as executor:
                            files_lst = list(
                                executor.map(
                                    lambda f: self.__get_file_data(f, ctx, csv), files
                                )
                            )
                        return files_lst

                    for f in files:

                        f_name = f.properties["Name"]
                        if target_file and target_file != f_name:
                            continue
                        data = self.__get_file_data(f, ctx, csv)
                        df = pd.concat([df, data], ignore_index=True)
                        target_folder_files.append(f_name)
                    break

            if target_file and not target_file in target_folder_files:
                print(f"File '{target_file}' not found in folder '{target_folder}'")
                return pd.DataFrame()

            return df

        except Exception as e:
            print("Problem getting folder content: ", e)

    def get_train_test_df(self, target_folder: str = "Analysis_files"):

        # Get train df
        train_df = self.get_train_df(target_folder)

        # Get test df
        test_df = self.get_test_df(target_folder)

        return train_df, test_df

    def get_test_df(self, target_folder: str = "Analysis_files"):
        test_df = pd.DataFrame()
        # Get test df
        for analysis_file in ANALYSIS_FILES_TEST:
            analysis_df = self.get_folder_contents(
                target_folder=target_folder,
                target_file=analysis_file,
            )
            test_df = pd.concat([test_df, analysis_df])

        return test_df

    def get_train_df(self, target_folder: str = "Analysis_files"):
        train_df = pd.DataFrame()
        # Get train df
        for analysis_file in ANALYSIS_FILES_TRAIN:
            analysis_df = self.get_folder_contents(
                target_folder=target_folder,
                target_file=analysis_file,
            )
            train_df = pd.concat([train_df, analysis_df])

        return train_df

    def download_test_folder(self, target_folder: str = "horses_jul_wins", csv=False):

        test_folder_names = self.get_folder_contents(
            target_folder=target_folder, only_folder_names=True, csv=csv
        )
        test_folder_files = self.get_folder_contents(
            target_folder=target_folder, only_folder_files=True, csv=csv
        )

        test_folder = list(zip(test_folder_names, test_folder_files))
        for f_name, f in test_folder:

            with open(f"test_folder/{f_name}", "wb") as local_file:
                local_file.write(f)

            print(f"Saved file successfully: {f_name}")

    def get_bsps(self, target_folder):
        bsp_df = self.get_folder_contents(
            target_folder=target_folder, target_file="bsp_df.csv"
        )
        return bsp_df


if __name__ == "__main__":
    onedrive = Onedrive(
        client_id=app_principal["client_id"],
        client_secret=app_principal["client_secret"],
        site_url=SITE_URL,
    )

    onedrive.download_test_folder()
