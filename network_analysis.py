from __future__ import division
import math, random, re
from collections import defaultdict, Counter, deque
from linear_algebra import dot, get_row, get_column, make_matrix, magnitude, scalar_multiply, shape, distance
from functools import partial

users = [
    { "id": 0, "name": "Hero" },
    { "id": 1, "name": "Dunn" },
    { "id": 2, "name": "Sue" },
    { "id": 3, "name": "Chi" },
    { "id": 4, "name": "Thor" },
    { "id": 5, "name": "Clive" },
    { "id": 6, "name": "Hicks" },
    { "id": 7, "name": "Devin" },
    { "id": 8, "name": "Kate" },
    { "id": 9, "name": "Klein" }
]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

#  Give each user a "friends list"
for user in users:
    user["friends"] = []

#  Populate list
for i, j in friendships:
    #  users[i] is the user whose id is "i"
    users[i]["friends"].append(users[j])        #  Add i as friend of j
    users[j]["friends"].append(users[i])        #  Add j as friend of i

##  BETWEENNESS CENTRALITY
def shortest_paths_from(from_user):
    """
    Shortest path from user
    """
    #  Dictionary from "user_id" to *all* shorted paths to the user
    shortest_paths_to = { from_user["id"] : [[]] }

    #  Queue of (previous user, next user) to check
    #  Starts with all pairs (from_user, friend_of_from_user)
    frontier = deque((from_user, friend) for friend in from_user["friends"])

    #  Keep going until queue is empty
    while frontier:

        prev_user, user = frontier.popleft()        #  Remove user show is 1st in queue
        user_id = user["id"]

        #  By way we are adding to queue, we already know some of the shortest
        #  paths to the user
        paths_to_prev = shortest_paths_to[prev_user["id"]]
        paths_via_prev = [path + [user_id] for path in paths_to_prev]

        #  Possible already know shortest path by now
        old_paths_to_here = shortest_paths_to.get(user_id, [])

        #  What is shortest path to here that has ben seen?
        if old_paths_to_here:
            min_path_length = len(old_paths_to_here[0])
        else:
            min_path_length = float('inf')

        #  Any new paths to here aren't too long
        new_paths_to_here = [path_via_prev
                             for path_via_prev in paths_via_prev
                             if len(path_via_prev) <= min_path_length
                             and path_via_prev not in old_paths_to_here]

        shortest_paths_to[user_id] = old_paths_to_here + new_paths_to_here

        #  Add new neighbors to frontier
        frontier.extend((user, friend)
                         for friend in user["friends"]
                         if friend["id"] not in shortest_paths_to)

    return shortest_paths_to

for user in users:
    user["shortest_paths"] = shortest_paths_from(user)

for user in users:
    user["betweenness_centrality"] = 0.0

for source in users:
    source_id = source["id"]
    for target_id, paths in source["shortest_paths"].iteritems():
        if source_id < target_id:       #  Don't double count!
            num_paths = len(paths)      #  How many shortest paths?
            contrib = 1 / num_paths     #  Contribution to centrality
            for path in paths:
                for id in path:
                    if id not in [source_id, target_id]:
                        users[id]["betweenness_centrality"] += contrib

##  CLOSENESS Centrality
