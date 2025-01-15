import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import tiktoken


def split_in_root_folders(input_path: str | Path) -> dict[str, list[str]]:
    """
    Splits the document into root folders and files.

    Each folder or file becomes a key in the returned dictionary. The values
    associated with these keys are lists containing the contents of the files
    within that folder or file
    (including subdirectories in the case of folders).

    :param input_path: Path to the input document to be split.
    :return: A dictionary where keys are folder or file names, and the values
             are lists of contents corresponding to those folders or files.
    """
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # This pattern captures the folder names between slashes,
    # i.e., /folder1/, /folder2/, or /file
    file_pattern = r'File: /([^/\n]+(?:\.[a-zA-Z0-9]+)?)'

    folder_names = re.findall(file_pattern, content)

    folders_content: dict[str, list[str]] = {
        key: [] for key in set(folder_names)
    }

    files = content.split('File: /')

    for file_content in files:
        folder_name = file_content.split('/')[0]
        if folder_name in folders_content.keys():
            folders_content[folder_name].append(file_content)

        file_name = file_content.split('\n')[0]
        if file_name in folders_content.keys():
            folders_content[file_name].append(file_content)

    return folders_content


def aggregate_files_by_token(
    data: dict[str, list[str]], max_tokens: int = 6000
) -> dict[str, list[str]]:
    """
    Aggregates files by token count, ensuring each aggregated string does not
    exceed a specified token limit. Splits files into smaller parts if
    necessary and adds context between parts.

    It also strip and replace the '=' in strings.

    :param data: A dictionary where the keys are the root folder, and the
                 values are lists of strings, where each record represents a
                 file inside the root directory to be aggregated (key).
    :param max_tokens: The maximum number of tokens allowed for each
                       aggregated string. Defaults to 6000.
    :return: A dictionary with the same keys as the input, but the values
             are lists of aggregated strings that comply with the token
             limit.
    """
    token_grouped_files: dict[str, list[str]] = defaultdict(list[str])

    for key, value in data.items():
        cumulative_string = ''

        for file in value:
            file_str = file.strip().replace('=', '')
            tokens = num_tokens_from_string(file_str)

            if tokens > max_tokens:
                if cumulative_string:
                    token_grouped_files[key].append(cumulative_string)
                    cumulative_string = ''

                parts = math.ceil(tokens / max_tokens)
                new_files = split_files_with_context(file_str, parts)
                token_grouped_files[key].extend(new_files)

            elif (
                num_tokens_from_string(file_str + cumulative_string)
                > max_tokens
            ):
                token_grouped_files[key].append(cumulative_string)
                cumulative_string = file_str

            else:
                cumulative_string += file_str

        if cumulative_string:
            token_grouped_files[key].append(cumulative_string)

    return token_grouped_files


def split_files_with_context(string: str, num_parts: int) -> list[str]:
    """
    Splits a string into multiple parts, ensuring each part has context by
    including a reference to the original file name and part number.

    :param string: The input string to be split into smaller parts.
                   Typically, this is a file's content.
    :type string: str
    :param num_parts: The number of parts to split the string into.
    :type num_parts: int
    :return: A list of strings, where each string represents a part of the
             original string, prefixed with the file name and part number.
    :rtype: list[str]
    """
    part_length = math.ceil(len(string) / num_parts)
    strings_split = [
        string[i : i + part_length] for i in range(0, len(string), part_length)
    ]

    file_name = string.split('\n')[0]
    names_to_concatenate = [
        file_name + f' - Parte ({n}/{num_parts})'
        for n in range(1, len(strings_split) + 1)
    ]

    strings_split = [
        '\n'.join([names_to_concatenate[i], strings_split[i]])
        for i in range(len(strings_split))
    ]

    return strings_split


def save_as_json(
    data: dict[str, list[str]],
    output_dir: str | Path,
    file_name: str = 'data.json',
) -> None:
    """
    Save a dictionary as a JSON file in the specified output directory.

    :param data: The dictionary to save as a JSON file.
                 Keys are strings, and values are lists of strings.
    :param output_dir: The path to the directory where the JSON file will
                       be saved. If the directory does not exist, it will be
                       created.
    :param file_name: The name of the JSON file to save. Defaults to
                      'data.json'.
    :return: None. The function saves the JSON file and prints the file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, file_name)

    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f'Saved: {path}')


def num_tokens_from_string(
    string: str, encoding_name: str = 'cl100k_base'
) -> int:
    """
    Returns the number of tokens in a text string.

    :param string: The input text string to be tokenized.
    :param encoding_name: The name of the encoding to use for tokenization.
                          Defaults to 'cl100k_base'.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))

    return num_tokens


if __name__ == '__main__':
    folder = 'data'
    input_file_name = 'source.txt'
    output_file_name = 'data_chunks.json'

    input_file = os.path.join(folder, input_file_name)

    split_data = split_in_root_folders(input_file)
    grouped_data = aggregate_files_by_token(split_data)

    save_as_json(grouped_data, folder, file_name=output_file_name)
