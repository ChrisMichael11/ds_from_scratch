from __future__ import division
from matplotlib import pyplot as plt
from collections import Counter, defaultdict

##########################
# FINDING KEY CONNECTORS #
##########################

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
    { "id": 9, "name": "Klein" },
    { "id": 10, "name": "Jen" }
]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

#  Give each user an empty list
for user in users:
    user["friends"] = []

#  Populate lists with friendships
for i, j in friendships:
    users[i]["friends"].append(users[j])  # Add i as a friend of j
    users[j]["friends"].append(users[i])  # Add j as friend of i

def number_of_friends(user):
    """
    How many friends does a user have?
    """
    return len(user["friends"])         # length of friends

total_connections = sum(number_of_friends(user) for user in users)
# print total_connections               # result = 24

num_users = len(users)                  # length of users list
# print num_users                       #  result = 11
avg_connections = total_connections / num_users
# print avg_connections                 #  result = 2.18

#  Create list of (user_id, number_of_friends)
num_friends_by_id = [(user["id"], number_of_friends(user)) for user in users]
# print num_friends_by_id

sorted(num_friends_by_id, key=lambda (user_id, num_friends): num_friends, reverse=True)
# print num_friends_by_id

################################
# DATA SCIENTISTS YOU MAY KNOW #
################################
def friends_of_friend_ids_bad(user):
    """
    Suggest friends of friends, 'foaf' = friend of a friend
    """
    return[foaf['id']
            for friend in user['friends']       #  For each of user's friends
            for foaf in friend['friends']]      #  Get each of user's friends

print friends_of_friend_ids_bad(users[0])       #  [0, 2, 3, 0, 1, 3]

#  There are lots of people connected through multiple connections...
print [friend["id"] for friend in users[0]["friends"]] # [1, 2]
print [friend["id"] for friend in users[1]["friends"]] # [0, 2, 3]
print [friend["id"] for friend in users[2]["friends"]] # [0, 1, 3]

def not_the_same(user, other_user):
    """
    Determines if users are not the same
    """
    return user["id"] != other_user["id"]

def not_friends(user, other_user):
    """
    other_user is not a friend if he's not in user['friends']
    if he's not_the_same as all people in user['friends']
    """
    return all(not_the_same(friend, other_user) for friend in user['friends'])

def friends_of_friend_ids(user):
    """
    returns list of friends of a user with their friends
    """
    return Counter(foaf["id"]
                for friend in user['friends']       #  For each friend
                for foaf in friend['friends']       #  Count "their" friends
                if not_the_same(user, foaf)         #  who aren't user
                and not_friends(user, foaf))        #  and aren't users friends

print friends_of_friend_ids(users[3])               #  Counter({0: 2, 5: 1})

interests = [
    (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
    (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
    (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
    (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
    (3, "statistics"), (3, "regression"), (3, "probability"),
    (4, "machine learning"), (4, "regression"), (4, "decision trees"),
    (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
    (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
    (6, "probability"), (6, "mathematics"), (6, "theory"),
    (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
    (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
    (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
    (9, "Java"), (9, "MapReduce"), (9, "Big Data")
]

def data_scientists_who_like(target_interest):
    """
    Find users with particular interest
    """
    return [user_id
            for user_id, user_interest in interests
            if user_interest == target_interest]
    #  Good, but must examine whole list for every search.
    #  Better off building index!!!

#  keys = interests, values = list of user_id with that interest
user_ids_by_interest = defaultdict(list)

for user_id, interest in interests:
    user_ids_by_interest[interest].append(user_id)

#  Another form of users to interests
#  keys are user_ids, values are lists of intereests for that user_id
interests_by_user_id = defaultdict(list)

for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)

def most_common_interests_with(user_id):
    """
    lists most common interests per given user
    """
    return Counter(interested_user_id
            for interest in interests_by_user_id[user["id"]]
            for interested_user_id in user_ids_by_interest[interest]
            if interested_user_id != user["id"])

###########################
# SALARIES AND EXPERIENCE #
###########################

salaries_and_tenures = [(83000, 8.7), (88000, 8.1),
                        (48000, 0.7), (76000, 6),
                        (69000, 6.5), (76000, 7.5),
                        (60000, 2.5), (83000, 10),
                        (48000, 1.9), (63000, 4.2)]

def make_chart_salaries_by_tenure():
    tenures = [tenure for salary, tenure in salaries_and_tenures]
    salaries = [salary for salary, tenure in salaries_and_tenures]
    plt.scatter(tenures, salaries)
    plt.xlabel("Years Experience")
    plt.ylabel("Salary")
    plt.show()

salary_by_tenure = defaultdict(list)

#  Keys are years, values are list of salaries for each year of tenure
for salary, tenure in salaries_and_tenures:
    salary_by_tenure[tenure].append(salary)

#  Keys are years value is average salary for that year of tenure
average_salary_by_tenure = {
    tenure : sum(salaries) / len(salaries)
    for tenure, salaries in salary_by_tenure.items()
}

def tenure_bucket(tenure):
    """
    bucket tenure
    """
    if tenure < 2:
        return "less than two"
    elif tenure < 5:
        return "between two and five"
    else:
        return "more than five"

#  Group salaries by bucket
#  Keys are tenure buckets, values are lists of salaries for the bucket
salary_by_tenure_bucket = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salary_by_tenure_bucket[bucket].append(salary)

#  Compute average salary for each Group
#  Keys are tenure buckets, values are average salary for that bucket
average_salary_by_bucket = {
    tenure_bucket : sum(salaries) / len(salaries)
    for tenure_bucket, salaries in salary_by_tenure_bucket.iteritems()
}
# print average_salary_by_bucket

#################
# PAID_ACCOUNTS #
#################
def predict_paid_or_unpaid(years_experience):
    if years_experience < 3.0: return "paid"
    elif years_experience < 8.5: return "unpaid"
    else:  return "paid"

######################
# TOPICS OF INTEREST #
######################
interests = [
        (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
        (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
        (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
        (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
        (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
        (3, "statistics"), (3, "regression"), (3, "probability"),
        (4, "machine learning"), (4, "regression"), (4, "decision trees"),
        (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
        (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
        (6, "probability"), (6, "mathematics"), (6, "theory"),
        (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
        (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
        (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
        (9, "Java"), (9, "MapReduce"), (9, "Big Data")
]

words_and_counts = Counter(word for user, interest in interests
                                for word in interest.lower().split())

if __name__ == "__main__":
    print
    print "######################"
    print "# FINDING KEY CONNECTORS"
    print "######################"
    print


    print "total connections", total_connections
    print "number of users", num_users
    print "average connections", total_connections / num_users
    print

    # create a list (user_id, number_of_friends)
    num_friends_by_id = [(user["id"], number_of_friends(user))
                         for user in users]

    print "users sorted by number of friends:"
    print sorted(num_friends_by_id,
                 key=lambda (user_id, num_friends): num_friends, # by number of friends
                 reverse=True)                                   # largest to smallest

    print
    print "######################"
    print "# DATA SCIENTISTS YOU MAY KNOW"
    print "######################"
    print


    print "friends of friends bad for user 0:", friends_of_friend_ids_bad(users[0])
    print "friends of friends for user 3:", friends_of_friend_ids(users[3])

    print
    print "######################"
    print "# SALARIES AND TENURES"
    print "######################"
    print

    print "average salary by tenure", average_salary_by_tenure
    print "average salary by tenure bucket", average_salary_by_bucket

    print
    print "######################"
    print "# MOST COMMON WORDS"
    print "######################"
    print

    for word, count in words_and_counts.most_common():
        if count > 1:
            print word, count
