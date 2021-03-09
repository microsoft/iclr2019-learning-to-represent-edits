# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from git import *
import sys, os

repos_folder = sys.argv[1]

for repo_folder in filter(lambda x: os.path.isdir(os.path.join(repos_folder, x)), os.listdir(repos_folder)):
    repo = Repo(path='.')

    for commit in list(repo.iter_commits()):
    if len(commit.parents) != 1:
        continue
        
    parent_commit = commit.parents[0]
        
    for diff_modified in parent_commit.diff(commit).iter_change_type('M'):
        if diff_modified.a_path.endswith('.cs') and diff_modified.b_path.endswith('.cs'):            
            assert diff_modified.a_blob.data_stream.read() != diff_modified.b_blob.data_stream.read()
            
            print('parent commit: %s' % parent_commit.hexsha)
            print('this commit: %s' % commit.hexsha)
            print(diff_modified.a_blob.path)
            print(diff_modified.a_blob.data_stream.read())
            print(diff_modified.b_blob.data_stream.read())
