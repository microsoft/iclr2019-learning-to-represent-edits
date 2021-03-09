# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import requests


def download(url: str) -> str:
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print("Downloading %s returned %s" % (url, response.status_code))
        return ''
