# Copyright 2023 Together Computer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""RedPajama: An Open-Source, Clean-Room 1.2 Trillion Token Dataset."""


import os
import json
import random
import datasets
import requests
import traceback

logger = datasets.logging.get_logger(__name__)
logger.level=1


_DESCRIPTION = """\
RedPajama is a clean-room, fully open-source implementation of the LLaMa dataset.
"""

# Function to save URLs to a specific file
def save_urls_to_file(subset, urls):
    filepath = f"urls/{subset}.txt"
    with open(filepath, "w") as file:
        for url in urls:
            file.write(url + "\n")

# Get all URLs.
url = 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt'
response = requests.get(url)
_ALL_URLS = response.content.decode('utf-8').split('\n')

# Subset specific URLs
_URL_LISTS = {
    "arxiv": "../urls/arxiv.txt",
    "book": "../urls/book.txt",
    "c4": "../urls/c4.txt",
    "common_crawl": "../urls/common_crawl.txt",
    "github": "../urls/github.txt",
    "stackexchange": "../urls/stackexchange.txt",
    "wikipedia": "../urls/wikipedia.txt",
}

# Create 'urls' directory if it doesn't exist
if not os.path.exists('urls'): os.makedirs('urls')

# Save URLs to subset-specific files
for subset, filepath in _URL_LISTS.items():
    urls_for_subset = [url for url in _ALL_URLS if subset in url]
    save_urls_to_file(subset, urls_for_subset)

_URL_BASE = 'https://data.together.xyz/redpajama-data-1T/v1.0.0'

_DATA_DIR = os.environ.get('RED_PAJAMA_DATA_DIR', None)

class RedPajama1TConfig(datasets.BuilderConfig):
    """BuilderConfig for RedPajama sample."""

    def __init__(self, *args, subsets, **kwargs):
        """BuilderConfig for RedPajama.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(RedPajama1TConfig, self).__init__(**kwargs)
        self.subsets = subsets


class RedPajama1T(datasets.GeneratorBasedBuilder):
    """RedPajama: Reproducing the LLaMA training dataset of over 1.2 trillion tokens. Version 1.0.0."""

    BUILDER_CONFIGS = [
        RedPajama1TConfig(
            name = 'default',
            subsets = list(_URL_LISTS.keys()),
            version=datasets.Version("1.0.0", ""),
            description="RedPajama1T",
        ),

        RedPajama1TConfig(
            name = 'arxiv',
            subsets = ['arxiv'],
            version=datasets.Version("1.0.0", ""),
            description="RedPajama1T arxiv subset",
        ),

        RedPajama1TConfig(
            name = 'book',
            subsets = ['book'],
            version=datasets.Version("1.0.0", ""),
            description="RedPajama1T book subset",
        ),

        RedPajama1TConfig(
            name = 'c4',
            subsets = ['c4'],
            version=datasets.Version("1.0.0", ""),
            description="RedPajama1T c4 subset",
        ),

        RedPajama1TConfig(
            name = 'common_crawl',
            subsets = ['common_crawl'],
            version=datasets.Version("1.0.0", ""),
            description="RedPajama1T common crawl subset",
        ),

        RedPajama1TConfig(
            name = 'github',
            subsets = ['github'],
            version=datasets.Version("1.0.0", ""),
            description="RedPajama1T github subset",
        ),

        RedPajama1TConfig(
            name = 'stackexchange',
            subsets = ['stackexchange'],
            version=datasets.Version("1.0.0", ""),
            description="RedPajama1T stackexchange subset",
        ),

        RedPajama1TConfig(
            name = 'wikipedia',
            subsets = ['wikipedia'],
            version=datasets.Version("1.0.0", ""),
            description="RedPajama1T wikipedia subset",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "meta": datasets.Value("string"),
                    "red_pajama_subset": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        url_lists = dl_manager.download_and_extract({
            subset: _URL_LISTS[subset] for subset in self.config.subsets
        })

        urls = []

        for subset, url_list in url_lists.items():
            with open(url_list, encoding="utf-8") as f:
                urls.extend([(subset, line.strip()) for line in f])

        # Select a single random URL from the combined list
        subset, url = random.choice(urls)
        logger.info(f"Selected subset: {subset}")
        logger.info(f"Selected URL: {url}")

        if _DATA_DIR is not None:
            url_prefix_slashes = len(_URL_BASE.split('/'))
            downloaded_files = {
                subset: [
                    os.path.join(_DATA_DIR, *url.split('/')[url_prefix_slashes:])
                ]
            }
        else:
            downloaded_files = dl_manager.download({subset: [url]})

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs = {
                    "files": {
                        subset: downloaded_files[subset]
                    }
                }
            )
        ]

    def _generate_examples(self, files):
        """This function returns the examples in the raw (text) form."""
        key = 0
        for subset in files:
            if subset == "common_crawl":
                import zstandard as zstd

                for path in files[subset]:
                    with zstd.open(open(path, "rb"), "rt", encoding="utf-8") as f:
                        for i, row in enumerate(f):
                            try:
                                data = json.loads(row)
                                text = data["text"]
                                del data["text"]
                                yield key, {
                                    "text": text,
                                    "meta": json.dumps(data),
                                    "red_pajama_subset": subset,
                                }
                                key += 1
                            except Exception as e:
                                print(f'Subset: {subset}')
                                print(f'Path: {path}')
                                print(f'Row: {row}')
                                traceback.print_exc()

                                raise e
            else:
                for path in files[subset]:
                    with open(path, encoding="utf-8") as f:
                        for i, row in enumerate(f):
                            try:
                                data = json.loads(row)
                                if "meta" not in data:
                                    text = data["text"]
                                    del data["text"]
                                    yield key, {
                                        "text": text,
                                        "meta": json.dumps(data),
                                        "red_pajama_subset": subset,
                                    }
                                else:
                                    yield key, {
                                        "text": data["text"],
                                        "meta": data["meta"],
                                        "red_pajama_subset": subset,
                                    }
                                key += 1
                            except Exception as e:
                                print(f'Subset: {subset}')
                                print(f'Path: {path}')
                                print(f'Row: {row}')
                                traceback.print_exc()

                                raise e

