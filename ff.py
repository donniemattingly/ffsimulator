import itertools
from collections import Counter
import numpy as np

from random import randint, shuffle, random
from collections import defaultdict
from operator import itemgetter, attrgetter
from statistics import mean, stdev

def gen_primes():
    """ Generate an infinite sequence of prime numbers.
    """
    # Maps composites to primes witnessing their compositeness.
    # This is memory efficient, as the sieve is not "run forward"
    # indefinitely, but only as long as required by the current
    # number being tested.
    #
    D = {}

    # The running integer that's checked for primeness
    q = 2

    while True:
        if q not in D:
            # q is a new prime.
            # Yield it and mark its first multiple that isn't
            # already marked in previous iterations
            #
            yield q
            D[q * q] = [q]
        else:
            # q is composite. D[q] is the list of primes that
            # divide it. Since we've reached q, we no longer
            # need it in the map, but we'll mark the next
            # multiples of its witnesses to prepare for larger
            # numbers
            #
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]

        q += 1


people = {
    'lizza': {
        'w': 7,
        'l': 4,
        'pf': 1690
    },
    'christian': {
        'w': 7,
        'l': 4,
        'pf':1489
    },
    'trevor': {
        'w': 6,
        'l': 5,
        'pf':1434
    },
    'donnie': {
        'w': 6,
        'l': 5,
        'pf':1551
    },
    'zach': {
        'w': 4,
        'l': 7,
        'pf':1480
    },
    'brian': {
        'w': 7,
        'l': 4,
        'pf':1543
    },
    'pankaj': {
        'w': 4,
        'l': 7,
        'pf':1508
    },
    'ken': {
        'w': 6,
        'l': 5,
        'pf':1491
    },
    'basil': {
        'w': 5,
        'l': 6,
        'pf':1708
    },
    'uday': {
        'w': 5,
        'l': 6,
        'pf':1546
    },
    'joshua': {
        'w': 5,
        'l': 6,
        'pf':1378
    },
    'jhahn': {
        'w': 4,
        'l': 7,
        'pf':1575
    }
}

for person in people:
	people[person]['name'] = person


def compute_outcome_number(winner, loser):
    return winner['win_prime'] * loser['lose_prime']


primes = gen_primes()
for person in people:
    win_prime = next(primes)
    lose_prime = next(primes)

    people[person]['win_prime'] = win_prime
    people[person]['lose_prime'] = lose_prime

num_to_outcome_map = {}

for c in itertools.combinations(people.values(), 2):
	if(c[0] != c[1]):
		a = compute_outcome_number(c[0], c[1])
		num_to_outcome_map[a] = {'w': c[0]['name'], 'l': c[1]['name']}

		b = compute_outcome_number(c[1], c[0])
		num_to_outcome_map[b] = {'w': c[1]['name'], 'l': c[0]['name']}



disallowed_factors = list(map(
        lambda x: x['win_prime'] * x['lose_prime'], people.values()))

def filter_invalid_games(outcome):
	return 0 in np.remainder(np.full((len(disallowed_factors)), sum(outcome)), disallowed_factors)

loser_nums = {}
winner_nums = {}

for person in people:
	loser_nums[person] = people[person]['lose_prime']
	winner_nums[person] = people[person]['win_prime']

def convert_bin_to_outcome(num, pow, week):
    outcome = []
    new_num = num
    for i in range(0, pow):
        outcome.append(week[i][new_num & 1])
        new_num = new_num >> 1

    print(num)
    return outcome



def get_week_outcomes(week):
    pow = len(week) - 1
    return map(lambda x: convert_bin_to_outcome(x, pow, week), range(0, 2**pow))


# def get_outcomes(schedule):


# def recursive_get_outcomes(cur_outcome, schedule):
#     if len(schedule) == 1:
#         return []
#     else:
#         head, *tail = schedule
#         return [cur_outcome + head[0], cur_outcome + head[0]]


week9 = [
    ('donnie', 'christian'),
    ('brian', 'lizza'),
    ('trevor', 'zach'),
    ('jhahn', 'joshua'),
    ('ken', 'uday'),
    ('pankaj', 'basil')]
week10 = [
    ('donnie', 'jhahn'),
    ('basil', 'ken'),
    ('uday', 'trevor'),
    ('brian', 'pankaj'),
    ('lizza', 'christian'),
    ('zach', 'joshua')]
week11 = [
    ('donnie', 'zach'),
    ('christian', 'brian'),
    ('jhahn', 'lizza'),
    ('pankaj', 'ken'),
    ('trevor', 'basil'),
    ('joshua', 'uday')]
week12 = [
    ('uday', 'donnie'),
    ('ken', 'trevor'),
    ('basil', 'joshua'),
    ('christian', 'pankaj'),
    ('brian', 'jhahn'),
    ('lizza', 'zach')]
week13 = [
    ('donnie', 'basil'),
    ('jhahn', 'christian'),
    ('zach', 'brian'),
    ('pankaj', 'trevor'),
    ('joshua', 'ken'),
    ('uday', 'lizza')
]

# outcomes = map(lambda x: [(x[0], x[1], 0), (x[0], x[1], 1)], schedule)
# flat_outcomes = [item for sublist in outcomes for item in sublist]
# permutations = itertools.permutations(outcomes)

schedule = [
    week12,
    week13
]

all_weeks = week9 + week10 + week11 + week12 + week13

# out_week9 = get_week_outcomes(week9)
# for outcome in out_week9:
#     count = Counter(outcome)
#     for k in people:
#         people[k]['wins'] = count[k]

# print(people)

# nums = [item for sublist in map(lambda x: [compute_outcome_number(
#     people[x[0]], people[x[1]]), compute_outcome_number(people[x[1]], people[x[0]])], week9) for item in sublist]

# combs = itertools.combinations(nums, 6)

# filtered = list(filter(lambda x: filter_invalid_games(x), combs))


def get_standings(season):
    results_arr = season.values()
    by_pf = sorted(results_arr, key=itemgetter('pf'), reverse=True)
    final = sorted(results_arr, key=itemgetter('w'), reverse=True)

    return final

def simulate_season(schedule, initial_state):
    wl = ['w', 'l']
    results = defaultdict(lambda: {'w': 0, 'l': 0})

    for week in schedule:
        for game in week:
            shuffle(wl)
            p1 = game[0]
            o1 = wl[0]

            p2 = game[1]
            o2 = wl[1]

            results[p1][o1] += 1
            results[p2][o2] += 1

    
    for name in results:
        person = results[name]
        initial = initial_state[name]

        person['w'] += initial['w']
        person['l'] += initial['l']
        person['pf'] = initial['pf']
        person['name'] = name


    return get_standings(results)

def simulate_season_account_for_points(schedule, initial_state):
    wl = ['w', 'l']
    results = defaultdict(lambda: {'w': 0, 'l': 0})

    for week in schedule:
        for game in week:
            p1 = game[0]
            p2 = game[1]

            p1_rate = initial_state[p1]['pf'] / (initial_state[p1]['pf'] + initial_state[p2]['pf'])
            r = random()

            o1 = 'w' if r < p1_rate else 'l'
            o2 = 'w' if o1 == 'l' else 'w'

            results[p1][o1] += 1
            results[p2][o2] += 1

    
    for name in results:
        person = results[name]
        initial = initial_state[name]

        person['w'] += initial['w']
        person['l'] += initial['l']
        person['pf'] = initial['pf']
        person['name'] = name


    return get_standings(results)


standings = {x:[] for x in people.keys()}
pf_standings = {x:[] for x in people.keys()}
num_trials = 10000

for i in range(0, num_trials):
    season = simulate_season(schedule, people)
    for j, person in enumerate(season):
        standings[person['name']].append(j + 1)
    
    pf_season = simulate_season_account_for_points(schedule, people)
    for j, person in enumerate(pf_season):
        pf_standings[person['name']].append(j + 1)
    
print('Coin toss for each game')
avg_standings = [(x[0], mean(x[1]), stdev(x[1])) for x in standings.items()]
sorted_avg_standings = sorted(avg_standings, key=itemgetter(1))
for s in sorted_avg_standings:
    print(f"{s[0].ljust(10)}: {s[1]:.2f} ± {s[2]:.2f}")

# print('')
# print('Accounting for points for')
# pf_avg_standings = [(x[0], mean(x[1]), stdev(x[1])) for x in pf_standings.items()]
# pf_sorted_avg_standings = sorted(pf_avg_standings, key=itemgetter(1))
# for s in pf_sorted_avg_standings:
#     print(f"{s[0].ljust(10)}: {s[1]:.2f} ± {s[2]:.2f}")