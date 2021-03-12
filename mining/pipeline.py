#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Usage:
    pipeline.py extract_revision_file [options] COMMENT_FILE_DIR
    pipeline.py extract_review [options] COMMENT_FILE_DIR
    pipeline.py extract_revision_from_repos [options] REPOS_DIR

Options:
    -h --help                  Show this screen.
    --show-all                 Instead of showing just the "acted-upon" comments, show all of them.
    --output=<filename>        The filename for the output html. [default: output.jsonl]
    --repo_list=<filename>     List of repos to extract commits from [default: None]
    --single_file_commit       Only collect single file commit
"""

import os, sys
import json
from collections import OrderedDict

from git import *
from docopt import docopt

from utils.dataloading import load_json_gz
from mining.change_detection import detect_contiguous_change, is_valid_change
from visualization.datasetviz import print_related_diff


def is_valid_comment_file(file_path: str) -> bool:
    # only get .cs file
    return file_path.endswith('.cs')


def extract_revision_files(args):
    comment_files = [file for file in os.listdir(args['COMMENT_FILE_DIR']) if file.endswith('.json.gz')]

    f_out = open(args['--output'], 'w', encoding='utf-8')
    revision_hash_set = set()

    for comment_file in comment_files:
        print('Processing %s' % comment_file, file=sys.stderr)
        comments = load_json_gz(os.path.join(args['COMMENT_FILE_DIR'], comment_file))

        for i, comment in enumerate(comments):
            if not comment['acted_upon']:
                continue

            if not is_valid_comment_file(comment['filepath']):
                continue

            revision_id = '_'.join([str(comment['pull_request_id']), comment['filepath']])

            if revision_id in revision_hash_set:
                continue

            print('\t writing one revision for [%s]' % comment['filepath'], file=sys.stderr)

            prev_file = comment['original_file']
            updated_file = comment['final_file']

            entry = OrderedDict(id=revision_id, prev_file=prev_file, updated_file=updated_file)
            f_out.write(json.dumps(entry) + '\n')

            revision_hash_set.add(revision_id)

    f_out.close()


def extract_revision_from_repos(args):
    repos_folder = args['REPOS_DIR']
    f_out = open(args['--output'], 'w', encoding='utf-8')

    total_commits = 0
    repo_folders = list(filter(lambda x: os.path.isdir(os.path.join(repos_folder, x)), os.listdir(repos_folder)))
    if args['--repo_list'] != 'None':
        valid_repos = [l.strip() for l in open(args['--repo_list'])]
        repo_folders = [folder for folder in repo_folders if folder in valid_repos]
    print('Processing %d repos' % len(repo_folders), file=sys.stderr)

    for repo_folder in repo_folders:
        print('Counting commits in repo %s' % repo_folder, file=sys.stderr)
        repo = Repo(path=os.path.join(repos_folder, repo_folder))

        try:
            repo_valid_commit_num = len([commit for commit in repo.iter_commits() if len(commit.parents) == 1])
            total_commits += repo_valid_commit_num
        except:
            print('Error in counting commits in repo %s' % repo_folder, file=sys.stderr)
            del repo_folders[repo_folders.index(repo_folder)]

    print('Total number of commits: %d' % total_commits, file=sys.stderr)

    for repo_folder in repo_folders:
        repo = Repo(path=os.path.join(repos_folder, repo_folder))
        print('Processing repo %s' % repo_folder, file=sys.stderr)

        for commit in list(repo.iter_commits()):
            if len(commit.parents) != 1:
                continue

            parent_commit = commit.parents[0]

            if args['--single_file_commit']:
                diffs = list(parent_commit.diff(commit))
                if len(diffs) == 1:
                    diff_modified = diffs[0]
                    if diff_modified.change_type == 'M' and diff_modified.a_path.endswith('.cs') and diff_modified.b_path.endswith('.cs'):
                        prev_file_content = diff_modified.a_blob.data_stream.read()
                        if prev_file_content:
                            prev_file_content = prev_file_content.decode("utf-8", errors="ignore")

                        updated_file_content = diff_modified.b_blob.data_stream.read()
                        if updated_file_content:
                            updated_file_content = updated_file_content.decode("utf-8", errors="ignore")

                        if prev_file_content and updated_file_content and prev_file_content != updated_file_content:
                            revision_id = '|'.join([repo_folder, str(commit.hexsha), diff_modified.a_blob.path])

                            print('\t writing one revision [%s]' % revision_id, file=sys.stderr)
                            entry = OrderedDict(id=revision_id,
                                                prev_file=prev_file_content,
                                                updated_file=updated_file_content,
                                                message=commit.message.strip())

                            f_out.write(json.dumps(entry) + '\n')
            else:
                for diff_modified in parent_commit.diff(commit).iter_change_type('M'):
                    if diff_modified.a_path.endswith('.cs') and diff_modified.b_path.endswith('.cs'):
                        prev_file_content = diff_modified.a_blob.data_stream.read()
                        if prev_file_content:
                            prev_file_content = prev_file_content.decode("utf-8", errors="ignore")

                        updated_file_content = diff_modified.b_blob.data_stream.read()
                        if updated_file_content:
                            updated_file_content = updated_file_content.decode("utf-8", errors="ignore")

                        if prev_file_content and updated_file_content and prev_file_content != updated_file_content:

                            revision_id = '|'.join([repo_folder, str(commit.hexsha), diff_modified.a_blob.path])

                            print('\t writing one revision [%s]' % revision_id, file=sys.stderr)
                            entry = OrderedDict(id=revision_id, prev_file=prev_file_content, updated_file=updated_file_content)

                            f_out.write(json.dumps(entry) + '\n')

    f_out.close()


def extract_review_data(args):
    comment_files = [file for file in os.listdir(args['COMMENT_FILE_DIR']) if file.endswith('.json.gz')]

    f_out = open(args['--output'], 'w', encoding='utf-8')

    for comment_file in comment_files:
        print('Processing %s' % comment_file, file=sys.stderr)
        comment_id = 0
        comments = load_json_gz(os.path.join(args['COMMENT_FILE_DIR'], comment_file))
        for comment in comments:
            comment_id += 1
            if not comment['acted_upon']:
                continue

            if not is_valid_comment_file(comment['filepath']):
                continue

            prev_context_chunk, prev_ctx_start, prev_ctx_end, \
            updated_context_chunk, updated_ctx_start, updated_ctx_end = print_related_diff(comment['original_file'],
                                                                                           comment['final_file'],
                                                                                           comment['position_in_file'])

            change_idx = detect_contiguous_change(prev_context_chunk, updated_context_chunk, num_contiguous_line=2)
            if change_idx:
                i1, i2, j1, j2 = change_idx
                prev_code_chunk = '\n'.join(prev_context_chunk[i1:i2])
                updated_code_chunk = '\n'.join(updated_context_chunk[j1:j2])
                change = OrderedDict(id='%s_%s_%s' % (comment['project'], comment['pull_request_id'], comment_id),
                                     prev_file=comment['original_file'],
                                     prev_linespan=(prev_ctx_start + i1, prev_ctx_start + i2 - 1),
                                     updated_file=comment['final_file'],
                                     updated_linespan=(updated_ctx_start + j1, updated_ctx_start + j2 - 1),
                                     prev_code_chunk=prev_code_chunk, updated_code_chunk=updated_code_chunk)

                if is_valid_change(change):
                    print('\t writing one change for [%s]' % comment['filepath'], file=sys.stderr)
                    change_entry = dict(pull_request_id=comment['pull_request_id'],
                                        link=comment['link'],
                                        project=comment['project'],
                                        change=change)

                    f_out.write(json.dumps(change_entry) + '\n')

    f_out.close()


if __name__ == '__main__':
    args = docopt(__doc__)
    if args['extract_review']:
        extract_review_data(args)
    elif args['extract_revision_file']:
        extract_revision_files(args)
    elif args['extract_revision_from_repos']:
        extract_revision_from_repos(args)
