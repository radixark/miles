from typing import Dict, Optional
import numpy as np
from Gym.environment import VerifiableEnvironment
import re

import copy
import enum
import itertools
import unittest

import numpy as np



"""Knight and Knave problems.

Each person can have the following (recursive) statements:
    - assertion: (telling-truth, i), (lying, i)
    - negation: (not, statement)
    - conjunction: (and, statement1, statement2), could support more than 2
    - disjunction: (or, statement1, statement2), could support more than 2
    - implication: (->, statement1, statement2)
    - equivalence: (<=>, statement1, statement2)

Please see the unit tests at the bottom on examples of how to use each API.

Original link: https://github.com/AlphaPav/mem-kk-logic/blob/main/data_prep/lib_kk.py
"""

####################################################################################
# Problem Solving
####################################################################################
def find_solution(statements):
  """Find solutions given a list of statements."""
  n_people = len(statements)
  single_statement = ('and',) + tuple(('<=>', ('telling-truth', i), statements[i])
                                      for i in range(len(statements)))
  # brute force
  solutions = []
  for assignments in itertools.product([True, False], repeat=n_people):
    if test_satisfiability(single_statement, assignments):
      solutions.append(assignments)

  return solutions


def test_satisfiability(statement, assignments):
  """Dumb recursive testing."""
  if statement[0] == 'telling-truth':
    return assignments[statement[1]]
  if statement[0] == 'lying':
    return not assignments[statement[1]]
  if statement[0] == 'not':
    return not test_satisfiability(statement[1], assignments)
  if statement[0] == 'and':
    return np.all([test_satisfiability(statement[i], assignments)
                   for i in range(1, len(statement))])
  if statement[0] == 'or':
    return np.any([test_satisfiability(statement[i], assignments)
                   for i in range(1, len(statement))])
  if statement[0] == '->':
    val1 = test_satisfiability(statement[1], assignments)
    val2 = test_satisfiability(statement[2], assignments)
    return (not val1) or val2
  if statement[0] == '<=>':
    val1 = test_satisfiability(statement[1], assignments)
    val2 = test_satisfiability(statement[2], assignments)
    return (val1 and val2) or ((not val1) and (not val2))
  raise KeyError(f'Unknown statement: {statement}')


####################################################################################
# Problem Sampling
####################################################################################
class KKProblemSampler:
  """Problem Sampler for Knight and Knave.

  Args:
    rand_seed: seed for random number generators.
    n_people: number of people for K&K problems.
    depth_constraint: the max depth of each person's statement. The depth refer to the level of
        recursion of operators such as 'and', 'or', etc. Increasing the depth would allow
        increasing the difficulty. Though currently the automatic formatting of the problems
        into nautral languages does not support depth more than 2.
    width_constraint: the max width (number of branches in operators such as 'and', 'or') of each
        person's statement.
  """

  def __init__(self, rand_seed: int, n_people: int, depth_constraint: int = 2, width_constraint: int = 2):
    self.rng = np.random.default_rng(rand_seed)
    self.rng_wrong = np.random.default_rng(rand_seed+1)
    self.n_people = n_people
    self.depth_constraint = depth_constraint
    self.width_constraint = width_constraint

  def sample(self):
    """Sample a single K&K problem."""
    statements = tuple(self._sample_statement(person_id, self.depth_constraint)
                       for person_id in range(self.n_people))
    return self._immutable_statements(statements)

  def sample_valid_problems(self, n_problems: int, max_retry: int = 1000,
                            skip_no_solution: bool = True, skip_multiple_solutions: bool = True):
    """Sample valid (has 1 unique solution) problems.

    Args:
      n_problems: how many problems to sample.
      max_retry: max number of retries per problem before giving up.
      skip_no_solution: skip problems without a valid solution.
      skip_multiple_solutions: skip problems with more than one solutions.

    Returns
      A list of problems, each a dict with keys 'statements' and 'solution'.
    """
    problems = []
    unique_statements = set()
    for i_problem in range(n_problems):
      for _ in range(max_retry):
        statements = self.sample()
        if statements in unique_statements:
          continue  # duplicated problem, retry
        solutions = find_solution(statements)
        if len(solutions) == 0 and skip_no_solution:
          continue  # retry
        if len(solutions) > 1 and skip_multiple_solutions:
          continue  # retry
        sol = solutions[0] if len(solutions) > 0 else None
        problems.append({'statements': statements, 'solution': sol,
                         'all_solutions': solutions})
        unique_statements.add(statements)
        break  # continue to next problem
      if i_problem + 1 < len(problems):
        raise RuntimeError(f'Failed to generate a valid problem after {max_retry} retries.')
    return problems

    def sample_flipped_solution(self, solution):
      length_of_solution = len(solution)
      # Randomly decide how many items to flip (at least one)
      num_to_perturb = self.rng_wrong.integers(1, length_of_solution)

      # Randomly choose indices to perturb
      indices_to_perturb = list(self.rng_wrong.choice(list(range(length_of_solution)), size=num_to_perturb, replace=False))
      
      # Create a new solution with perturbed values
      perturbed_solution = tuple(
          not solution[i] if i in indices_to_perturb else solution[i]
          for i in range(length_of_solution)
      )
      return perturbed_solution


  def sample_invalid_problems(self, n_problems: int, max_retry: int = 1000,
                            skip_no_solution: bool = True, skip_multiple_solutions: bool = True):
    """Sample valid (has 1 unique solution) problems and then perturb the solution.

    Args:
      n_problems: how many problems to sample.
      max_retry: max number of retries per problem before giving up.
      skip_no_solution: skip problems without a valid solution.
      skip_multiple_solutions: skip problems with more than one solutions.

    Returns
      A list of problems, each a dict with keys 'statements' and 'solution'.
    """
    problems = []
    unique_statements = set()
    for i_problem in range(n_problems):
      for _ in range(max_retry):
        statements = self.sample()
        if statements in unique_statements:
          continue  # duplicated problem, retry
        solutions = find_solution(statements)
        if len(solutions) == 0 and skip_no_solution:
          continue  # retry
        if len(solutions) > 1 and skip_multiple_solutions:
          continue  # retry
        sol = solutions[0] if len(solutions) > 0 else None
        ## perturbed
        perturbed_sol=self.sample_flipped_solution(sol)
        problems.append({'statements': statements, 'solution': perturbed_sol,
                         'all_solutions': [perturbed_sol]})
        unique_statements.add(statements)
        break  # continue to next problem
      if i_problem + 1 < len(problems):
        raise RuntimeError(f'Failed to generate a valid problem after {max_retry} retries.')
    return problems


  def perturb_problems(self, problems, max_retry: int = 1000, perturb_type: str = 'statement',
                       num_perturb: int = 1):
    """Perturb the problems (generated by this sampler).

    The perturbed problems will change in one place, and is guaranteed to have a different
    solution. The 'leaf' perturbation type allows "small" perturbation, but it will have a
    high chance of not able to generate valid perturbations when n_people is small (i.e. all
    the single-step perturbations do not lead to a valid solution). One potential solution is
    to enable `allow_failure` and filter out invalid ones (marked as None).

    Args:
      problems: a list of problems generated by this sampler.
      max_retry: max number of retries to generate an alternative and valid problem.
      perturb_type: 'leaf' means perturbing only a random leaf node (i.e. not compond statements);
          'statement' means change the entire statement from a random person.
      num_perturb: number of perturbations to generate. Note the actual returned perturbations
          might be fewer than this number (or even an empty list), if max_retry is exhausted.

    Returns:
      A list of perturbed problems.
    """
    return [self._perturb_problem(p, max_retry=max_retry, perturb_type=perturb_type, num_perturb=num_perturb)
            for p in problems]

  def _perturb_problem(self, problem, max_retry: int, perturb_type: str, num_perturb: int):
    assert len(problem['statements']) == self.n_people  # make sure parameters match
    results_set = set()
    results_list = []
    for _ in range(max_retry):
      statements = self._copy_statements_as_mutable(problem['statements'])
      if perturb_type == 'statement':
        person = self.rng.integers(0, self.n_people)
        statements[person] = self._sample_statement(person, depth_constraint=self.depth_constraint)
      elif perturb_type == 'leaf':
        person = self.rng.integers(0, self.n_people)
        idx = person
        container = statements
        while not self._is_leaf_node(container[idx]):
          container = container[idx]
          idx = self.rng.integers(1, len(container))
        assert self._is_leaf_node(container[idx])
        # set depth_constraint to 1 to only sample new leaf node
        container[idx] = self._sample_statement(person, depth_constraint=1)

      statements = self._immutable_statements(statements)
      if len(set([statements, problem['statements']])) <= 1:
        continue  # perturbation is identical to the original, retry

      solutions = find_solution(statements)
      if len(solutions) != 1:
        continue  # Not single unique solution, retry

      if len(set([solutions[0], problem['solution']])) <= 1:
        continue  # solution does not change after perturbation, retry

      if statements in results_set:
        continue  # duplicate perturbation, retry

      results_set.add(statements)
      results_list.append({'statements': statements, 'solution': solutions[0]})
      if len(results_list) >= num_perturb:
        break
    
    if len(results_list)==0:
      return [None]

    return results_list

  def _copy_statements_as_mutable(self, statements):
    """Make a deep copy of the statements of a problem, turning the tuples into (mutable) lists."""
    statements = copy.deepcopy(statements)
    def _make_mutable(x):
      if isinstance(x, tuple):
        return [_make_mutable(child) for child in x]
      return x
    return [_make_mutable(s) for s in statements]

  def _immutable_statements(self, mutable_statements):
    """Change list back to tuples."""
    def _make_immutable(x):
      if isinstance(x, (list, tuple)):
        return tuple(_make_immutable(child) for child in x)
      if isinstance(x, np.str_):
        return str(x)
      if isinstance(x, np.int64):
        return int(x)
      return x
    return tuple(_make_immutable(s) for s in mutable_statements)

  def _is_leaf_node(self, statement):
    if statement[0] in ['telling-truth', 'lying']:
      return True
    return False

  def _sample_statement(self, person_id: int, depth_constraint: int):
    """Sample a single statement."""
    dice = self.rng.integers(0, 6)
    if depth_constraint == 1 or dice == 0:
      while True:
        knight_or_knave = self.rng.choice(['telling-truth', 'lying'])
        person = self.rng.integers(0, self.n_people)
        if not (knight_or_knave == 'lying' and person == person_id):
          # avoid the trivially unsatisfiable statement
          return (knight_or_knave, person)

    if dice == 1:
      return ('not', self._sample_statement(person_id, depth_constraint-1))
    if dice in [2, 3]:
      operator = ['and', 'or'][dice-2]
      n_substatements = self.rng.integers(2, self.width_constraint+1)

      return (operator,) + self._sample_substatements(person_id, depth_constraint, n_substatements)
    if dice in [4, 5]:
      operator = ['->', '<=>'][dice-4]
      return (operator,) + self._sample_substatements(person_id, depth_constraint, 2)

  def _sample_substatements(self, person_id: int, depth_constraint: int, count: int, dedup: bool = True):
    """Sample substatements for an operator.

    Args:
      person_id: the id of the person making the statements.
      depth_constraint: the maximum depth of substatements.
      count: number of substatements to generate.
      dedup: if True, avoid duplicated substatements.
    """
    sub_statements = []
    dedup_set = set()
    while True:
      stmt = self._sample_statement(person_id, depth_constraint-1)
      if dedup:
        if stmt in dedup_set:
          continue
        dedup_set.add(stmt)

      sub_statements.append(stmt)
      if len(sub_statements) == count:
        break
    return tuple(sub_statements)


####################################################################################
# Problem Formatting in natural language
####################################################################################
COMMON_NAMES = ['Emma', 'Liam', 'Olivia', 'Noah', 'Ava', 'Ethan', 'Sophia',
                'Mason', 'Isabella', 'William', 'Mia', 'James', 'Charlotte',
                'Benjamin', 'Amelia', 'Lucas', 'Harper', 'Henry', 'Evelyn',
                'Alexander', 'Abigail', 'Michael', 'Emily', 'Daniel', 'Elizabeth',
                'Jacob', 'Sofia', 'Logan', 'Avery', 'Jackson', 'Ella', 'Sebastian',
                'Scarlett', 'Jack', 'Grace', 'Aiden', 'Chloe', 'Owen', 'Victoria',
                'Samuel', 'Riley', 'Matthew', 'Aria', 'Joseph', 'Lily', 'Luke',
                'Aurora', 'David', 'Zoey', 'Oliver', 'Penelope']
UNCOMMON_NAMES = [
    'Zephyr', 'Elowen', 'Caspian', 'Isolde', 'Osiris', 'Vesper', 'Thaddeus', 'Ondine',
    'Lysander', 'Xanthe', 'Oberon', 'Calliope', 'Leander', 'Eulalia', 'Florian', 'Forsythe',
    'Nephele', 'Peregrine', 'Ianthe', 'Lazarus', 'Elodie', 'Cillian', 'Ottoline', 'Evander',
    'Saffron', 'Caius', 'Zora', 'Cyprian', 'Amaryllis', 'Theron', 'Perdita', 'Ignatius',
    'Zephyrine', 'Balthazar', 'Melisande', 'Zinnia', 'Sylvester', 'Cosima', 'Leocadio',
    'Percival', 'Oceane', 'Evanthe', 'Zenobia', 'Eurydice', 'Quillan', 'Aeronwen',
    'Thorsten', 'Xiomara', 'Zephyrus', 'Ysolde'
]

KNIGHT_KNAVE_PAIRS = [
    # NOTE: we simply add 's' to make plural, so be careful when choosing words
    ['a pioneer', 'a laggard'],
    ['a saint', 'a sinner'],
    ['a hero', 'a villain'],
    ['an angel', 'a devil'],
    ['an altruist', 'an egoist'],
    ['a sage', 'a fool'],
]
PREFIX = ('A very special island is inhabited only by {knight}s and {knave}s. ' +
          '{Knight}s always tell the truth, and {knave}s always lie. ')
POSTFIX = 'So who is {a_knight} and who is {a_knave}?'
TEMPLATES = [  
    '{name} said that {content}.',
    '{name} told you that {content}.',
    '{name} said, "{content}."',
    '{name} stated, "{content}".',
    'According to {name}, "{content}".',
    '''In {name}'s words: "{content}".''',
    '{name} remarked, "{content}".',
    '"{content}," {name} declared.',
    '{name} was heard saying, "{content}".',
    '{name} expressed that {content}.',
    '"{content}" - {name}.',
    'As {name} put it, "{content}".',
    '{name} asserted: "{content}".',
    '"{content}," {name} mentioned.',
    '{name} commented, "{content}".',
    'In a statement by {name}: "{content}".',
    '{name} noted, "{content}".',
    '"{content}," {name} claimed.',
]


class KKProblemFormatter:

  def __init__(self, rand_seed, problem):
    self.rng = np.random.default_rng(rand_seed)
    self.rng_perturb = np.random.default_rng(rand_seed+1)
    self.problem = problem

  def format_problem(self, random_names=True, random_saying_template=True,
                     random_knight_knave_pairs=False,
                     flip_knight_knave_pair=False, uncommon_name=False, reorder_statement=False):
    statements = copy.deepcopy(self.problem['statements'])

    n_people = len(statements)
    names = COMMON_NAMES[:n_people]
    if random_names:
      if uncommon_name==False:
        names = list(self.rng.choice(COMMON_NAMES, size=n_people, replace=False))
      else:
        names = list(self.rng.choice(UNCOMMON_NAMES, size=n_people, replace=False))
    names = [str(x) for x in names]  # convert np.str_ to str

    knight_knave = ['a knight', 'a knave']
    if random_knight_knave_pairs:
      knight_knave = self.rng.choice(KNIGHT_KNAVE_PAIRS) 
    knight_knave = [str(x) for x in knight_knave]  # convert np.str_ to str

    if flip_knight_knave_pair:
      knight_knave = knight_knave[::-1]

    knight_knave = {'knight': knight_knave[0].split()[1],
                    'knave': knight_knave[1].split()[1],
                    'a_knight': knight_knave[0], 'a_knave': knight_knave[1]}
    knight_knave['Knight'] = knight_knave['knight'].capitalize()
    knight_knave['Knave'] = knight_knave['knave'].capitalize()

    text = PREFIX.format(**knight_knave)
    text += f'You meet {n_people} inhabitants: '
    text += ', '.join(names[:-1]) + ', and ' + names[-1] + '.'

    text_statements=[]
    for i, stmt in enumerate(statements):
      tmpl = TEMPLATES[0]
      if random_saying_template:
        tmpl = self.rng.choice(TEMPLATES)

      content = format_statement(names, knight_knave, stmt)
      text_statements.append(' ' + tmpl.format(name=names[i], content=content))
      # text += ' ' + tmpl.format(name=names[i], content=content)
    
    if reorder_statement:
      original_order = list(range(n_people))
      # Copy the original list
      shuffled_order = original_order.copy()

      # Shuffle until it's different from the original
      while True:
          self.rng_perturb.shuffle(shuffled_order)
          if shuffled_order != original_order:
              break
      for i in shuffled_order:
          text += text_statements[i]
    else:
      text += ''.join(text_statements)

    text += ' ' + POSTFIX.format(**knight_knave)
    if self.problem['solution'] is None:
      solution_text = 'No valid solution exists.'
    else:
      solution_stmts = []
      for name, indicator in zip(names, self.problem['solution']):
        if indicator:
          solution_stmts.append(name + ' is ' + knight_knave['a_knight'])
        else:
          solution_stmts.append(name + ' is ' + knight_knave['a_knave'])
      solution_text = ', '.join(solution_stmts[:-1]) + ', and ' + solution_stmts[-1] + '.'
    return {'quiz': text, 'names': names, 'knight_knave': knight_knave,
            'solution': self.problem['solution'],
            'solution_text': solution_text}


# TODO: currently we do not support formatting of problems with depth more than
# 2. We may need to use LLM or think more about what would be the best way
# to format complicated recursive statements.
def format_knight_knave(names, knight_knave, statement, negation=False):
  assert statement[0] in ('telling-truth', 'lying')
  text = names[statement[1]] + ' is '
  if negation:
    text += 'not '
  text += {'telling-truth': knight_knave['a_knight'],
           'lying': knight_knave['a_knave']}[statement[0]]
  return text


def format_statement(names, knight_knave, statement):
  if statement[0] == 'not':
    return format_knight_knave(names, knight_knave, statement[1], negation=True)
  if statement[0] in ['and', 'or']:
    text = (' ' + statement[0] + ' ').join(
        format_knight_knave(names, knight_knave, sub_stmt) for sub_stmt in statement[1:])
    return text
  if statement[0] == '->':
    return ('If ' + format_knight_knave(names, knight_knave, statement[1]) + ' then ' +
            format_knight_knave(names, knight_knave, statement[2]))
  if statement[0] == '<=>':
    return (format_knight_knave(names, knight_knave, statement[1]) + ' if and only if ' +
            format_knight_knave(names, knight_knave, statement[2]))
  return format_knight_knave(names, knight_knave, statement)


####################################################################################
# Chain of Thoughts
####################################################################################
def generate_chain_of_thoughts(statements, dynamic_person_order: bool = True):
  """Generate reasoning steps that can solve the problem.

  Args:
    statements: the statements of the K&K problem.
    dynamic_person_order: if False, it will always go through the list of person in the original order. If True,
      it will use a more "natural" order. For example, if person1 mention person5 and person4, then the engine will
      check person5 and person4 next, instead of checking person2 next.
  """
  n_people = len(statements)
  tape = []
  assignments = [None] * n_people
  options = {p: [False, True] for p in range(n_people)}
  persons_to_consider = tuple(range(n_people))
  p_cursor = 0
  while True:
    if p_cursor >= n_people:
      tape.append(('success', {'assignments': tuple(assignments)}))
      break

    if not options[persons_to_consider[p_cursor]]:
      exhausted = []
      while p_cursor >= 0 and not options[persons_to_consider[p_cursor]]:
        options[persons_to_consider[p_cursor]] = [False, True]
        assignments[persons_to_consider[p_cursor]] = None
        exhausted.append(persons_to_consider[p_cursor])
        p_cursor -= 1
      if p_cursor >= 0:
        tape.append(('reconsider', {'person': persons_to_consider[p_cursor], 'exhausted': exhausted}))
      else:
        # we have exhausted all options
        tape.append(('fail',))
        break

    person = persons_to_consider[p_cursor]
    assignments[person] = options[person].pop()
    result, stmt_id = can_be_falsified_v2(statements, assignments)
    if result:
      tape.append(('proposal', {'person': person, 'assignment': assignments[person],
                                'outcome': 'ok'}))
      # re-order the next people to consider based on who is mentioned in the current statement
      mentioned_people = _find_mentioned_people(statements[person])
      p_cursor += 1
      persons_to_consider = persons_to_consider[:p_cursor] + _reorder_people_sequence(
          persons_to_consider[p_cursor:], mentioned_people)
    else:
      tape.append(('proposal', {'person': person, 'assignment': assignments[person],
                                'outcome': 'conflict', 'conflict_statement': (stmt_id, assignments[stmt_id])}))
  return tape


def _find_mentioned_people(statement):
  """Find the id of people mentioned in the statement."""
  if statement[0] in ['lying', 'telling-truth']:
    return [statement[1]]
  if statement[0] in ['not', 'and', 'or', '->', '<=>']:
    return sum([_find_mentioned_people(s) for s in statement[1:]], [])
  raise KeyError(f'Unknown statement: {statement}')


def _reorder_people_sequence(remaining_people, mentioned_people):
  """Reorder the remaining people by brining the mentioned ones to the front."""
  # dedup and keep order
  set_uniq_mention = set()
  list_uniq_mention = []
  for p in mentioned_people:
    if p not in set_uniq_mention:
      set_uniq_mention.add(p)
      list_uniq_mention.append(p)

  for p in reversed(mentioned_people):
    if not p in remaining_people:
      continue
    idx = remaining_people.index(p)
    remaining_people = (p,) + remaining_people[:idx] + remaining_people[idx+1:]
  return remaining_people


def can_be_falsified_v2(statements, assignments):
  """Test falsifiability of partial assignment (v2).

  This version enumerate all possible remaining assignments. This is less efficient than v1. But v1 has
  the potential issue that it cannot easily detect self contradictory statement such as
  `('<=>', ('lying', 4), ('telling-truth', 4))` when the person 4's assignment is undecided yet.
  """
  n_people = len(statements)
  remap = [i for i, x in enumerate(assignments) if x is None]
  n_unassigned = len(remap)

  for p_idx in range(n_people):
    if assignments[p_idx] is None:
      continue
    p_statement = statements[p_idx]
    if not assignments[p_idx]:
      p_statement = ('not', p_statement)
    has_solution = False

    for proposal in itertools.product([True, False], repeat=n_unassigned):
      new_assignments = copy.copy(assignments)
      for i, x in zip(remap, proposal):
        new_assignments[i] = x
      if test_satisfiability(p_statement, new_assignments):
        has_solution = True
        break
    if not has_solution:
      return (False, p_idx)  # this person's statement cannot be satisfied

  return (True, None)


class TruthOrWhatever(enum.Enum):
  FALSE = 0
  TRUE = 1
  WHATEVER = 2

  @classmethod
  def from_bool(cls, val: bool):
    if val:
      return cls.TRUE
    else:
      return cls.FALSE

  def f_not(self):
    if self == self.TRUE:
      return self.FALSE
    if self == self.FALSE:
      return self.TRUE
    return self.WHATEVER

  def f_and(self, other):
    if self == self.WHATEVER or other == self.WHATEVER:
      return self.WHATEVER
    if self == self.TRUE:
      return self.from_bool(other == self.TRUE)
    return self.FALSE

  def f_or(self, other):
    if self == self.WHATEVER or other == self.WHATEVER:
      return self.WHATEVER
    if self == self.FALSE:
      return self.from_bool(other == self.TRUE)
    return self.TRUE


def can_be_falsified(statements, assignments):
  """Test if the (partial) assignment can be falsified."""
  def _test(stmt) -> TruthOrWhatever:
    if stmt[0] in ['telling-truth', 'lying'] and assignments[stmt[1]] is None:
      return TruthOrWhatever.WHATEVER
    if stmt[0] == 'telling-truth':
      return TruthOrWhatever.from_bool(assignments[stmt[1]] is True)
    if stmt[0] == 'lying':
        return TruthOrWhatever.from_bool(assignments[stmt[1]] is False)
    if stmt[0] == 'not':
      return _test(stmt[1]).f_not()
    if stmt[0] == 'and':
      val = _test(stmt[1])
      for sub_stmt in stmt[2:]:
        val = val.f_and(_test(sub_stmt))
      return val
    if stmt[0] == 'or':
      val = _test(stmt[1])
      for sub_stmt in stmt[2:]:
        val = val.f_or(_test(sub_stmt))
      return val
    if stmt[0] == '->':
      val1 = _test(stmt[1])
      val2 = _test(stmt[2])
      return val1.f_not().f_or(val2)
    if stmt[0] == '<=>':
      val1 = _test(stmt[1])
      val2 = _test(stmt[2])
      return val1.f_and(val2).f_or(val1.f_not().f_and(val2.f_not()))
    raise KeyError(f'Unknown statement: {stmt}')

  for i, (stmt, assmt) in enumerate(zip(statements, assignments)):
    if assmt is None:
      # this person's claim does not matter
      continue
    if assmt and _test(stmt) == TruthOrWhatever.FALSE:
      return (False, i)
    if not assmt and _test(stmt) == TruthOrWhatever.TRUE:
      return (False, i)
  return (True, None)


def format_chain_of_thoughts(problem, formatted_problem, tape,
                             repeat_claim_for_assumption: bool = True,
                             repeat_claim_for_contradiction: bool = False):
  """Format generate chain-of-thoughts in natural language.

  Repeating the claim makes it a bit more natural, but also increas the number of tokens needed to handle.

  Args:
    problem: the K&K problem.
    formatted_problem: the formatted results of the K&K problem.
    tape: the generated chain of thoughts.
    repeat_claim_for_assumption: whether to repeat each person's claim after we assuming they are a knight or knave.
    repeat_claim_for_contradiction: whether to repeat the contradicted claim when a contradiction is found.

  Returns:
    (header, [step1, step2, ...], footer). The footer contains a conclusion of success or failure. Note the final
    solution is not included in the footer. If needed, problem['solution_text'] can be appended here.
  """
  format_dict = copy.copy(formatted_problem['knight_knave'])
  n_person = len(problem['statements'])
  for p in range(n_person):
    format_dict[f'P{p}'] = formatted_problem['names'][p]

  header = "Let's think step by step, by considering whether each person is lying and if that leads to contradiction."
  steps = []
  for step in tape[:-1]:  # last step is fail / success
    if step[0] == 'proposal':
      t_person = '{P' + str(step[1]['person']) + '}'
      t_assignment = '{a_knight}' if step[1]['assignment'] else '{a_knave}'
      if step[1]['outcome'] == 'ok':
        text = 'Assume ' + t_person + ' is ' + t_assignment + '.'
        if repeat_claim_for_assumption:
          t_claim = format_statement(formatted_problem['names'], formatted_problem['knight_knave'],
                                     problem['statements'][step[1]['person']])
          text += ' No contradiction is found in their '
          if not step[1]['assignment']:
            text += 'false '
          text += 'claim that ' + t_claim + '.'
      elif step[1]['outcome'] == 'conflict':
        conflict_p, conflict_assignment = step[1]['conflict_statement']
        text = t_person + ' cannot be ' + t_assignment + ', because this would contradict the '
        if not conflict_assignment:
          text += 'false '
        text += 'claim of '
        if conflict_p == step[1]['person']:
          text += 'their own'
        else:
          text += '{P' + str(conflict_p) + '}'
        if repeat_claim_for_contradiction:
          t_claim = format_statement(formatted_problem['names'], formatted_problem['knight_knave'],
                                     problem['statements'][conflict_p])
          text += ' that ' + t_claim + '.'
        else:
          text += '.'
      else:
        raise KeyError(f'Unknown outcome for CoT step: {step}')
      steps.append(text)
    elif step[0] == 'reconsider':
      text = 'We have exhausted all possibilities for '
      t_exhausted = ['{P' + str(p_idx) + '}' for p_idx in step[1]['exhausted']]
      assert len(t_exhausted) > 0
      if len(t_exhausted) == 1:
        text += t_exhausted[0]
      elif len(t_exhausted) == 2:
        text += ' and '.join(t_exhausted)
      else:
        t_exhausted[-1] = 'and ' + t_exhausted[-1]
        text += ', '.join(t_exhausted)
      text += ', so let us go back and reconsider {P' + str(step[1]['person']) + '}.'
      steps.append(text)
    else:
      raise KeyError(f'Unknown CoT step: {step}')

  if tape[-1][0] == 'success':
    footer = 'This leads to a feasible solution.'
  elif tape[-1][0] == 'fail':
    footer = 'All the configurations lead to contradictions.'
  else:
    raise KeyError(f'Expect success or fail, but get {tape[-1]}')

  steps = [x.format(**format_dict) for x in steps]
  return (header, steps, footer)


####################################################################################
# Unit Testing
####################################################################################
class TestKK(unittest.TestCase):

  def test_find_solution(self):
    statements = (
        ('lying', 1),
        ('and', ('telling-truth', 0), ('telling-truth', 1))
    )
    sol = find_solution(statements)
    self.assertEqual(sol, [(True, False)])

  def test_sample_problems(self):
    n_people = 3
    n_problems = 5
    problem_sampler = KKProblemSampler(1234, n_people=n_people)
    problems = problem_sampler.sample_valid_problems(n_problems)
    self.assertEqual(len(problems), n_problems)
    for problem in problems:
      self.assertEqual(set(problem.keys()), set(['statements', 'solution', 'all_solutions']))
      self.assertEqual(len(problem['statements']), n_people)

  def test_format_problems(self):
    problem_sampler = KKProblemSampler(1234, n_people=3)
    problems = problem_sampler.sample_valid_problems(20, skip_no_solution=False)

    for problem in problems:
      formatter = KKProblemFormatter(rand_seed=1234, problem=problem)
      formatted_results = formatter.format_problem()
      self.assertIn('quiz', formatted_results)
      self.assertIn('names', formatted_results)
      self.assertIn('solution', formatted_results)
      self.assertIn('solution_text', formatted_results)
      if problem['solution'] is None:
        self.assertEqual(formatted_results['solution_text'], 'No valid solution exists.')

  def test_perturb_problems(self):
    n_people = 4
    n_perturb = 3
    problem_sampler = KKProblemSampler(1234, n_people=n_people)
    problems = problem_sampler.sample_valid_problems(5)
    for perturb_type in ['statement', 'leaf']:
      perturbed_problems = problem_sampler.perturb_problems(problems, perturb_type=perturb_type, num_perturb=n_perturb)
      self.assertEqual(len(problems), len(perturbed_problems))
      for p1, p2_list in zip(problems, perturbed_problems):
        self.assertEqual(len(p2_list), n_perturb)  # note this can actual fail, esp for small n_people
        self.assertNotEqual(p1['solution'], p2_list[0]['solution'])
        n_stmt_diff = 0
        for s1, s2 in zip(p1['statements'], p2_list[0]['statements']):
          if s1 != s2:
            n_stmt_diff += 1
        self.assertEqual(n_stmt_diff, 1)  # exactly 1 statement is different

  def test_chain_of_thoughts(self):
    n_people = 5
    n_problems = 120
    problem_sampler = KKProblemSampler(1234, n_people=n_people)
    problems = problem_sampler.sample_valid_problems(n_problems, skip_no_solution=False)
    for p in problems:
      for dynamic_person_order in [False, True]:
        tape = generate_chain_of_thoughts(p['statements'], dynamic_person_order=dynamic_person_order)
        if p['solution'] is None:
          self.assertTupleEqual(tape[-1], ('fail',))
        else:
          self.assertEqual(tape[-1][0], ('success'))
          self.assertTupleEqual(tape[-1][1]['assignments'], p['solution'])

  def test_chain_of_thoughts_regression(self):
    # Regression test: NOTE the correct answer is not unique and it can change when the CoT generator code
    # is changed. So the failure of this test does not necessarily mean the code is incorrect. If the code
    # is changed and verified to be correct, this test can be updated with the new target outputs.
    statements = (('and', ('telling-truth', 2), ('lying', 3)),
                  ('telling-truth', 2),
                  ('<=>', ('lying', 4), ('telling-truth', 4)),
                  ('and', ('lying', 2), ('lying', 4)),
                  ('lying', 2))
    expected_tape = [
        ('proposal', {'person': 0, 'assignment': True, 'outcome': 'ok'}),
        ('proposal',
          {'person': 2,
          'assignment': True,
          'outcome': 'conflict',
          'conflict_statement': (2, True)}),
        ('proposal',
          {'person': 2,
          'assignment': False,
          'outcome': 'conflict',
          'conflict_statement': (0, True)}),
        ('reconsider', {'person': 0, 'exhausted': [2]}),
        ('proposal', {'person': 0, 'assignment': False, 'outcome': 'ok'}),
        ('proposal',
          {'person': 2,
          'assignment': True,
          'outcome': 'conflict',
          'conflict_statement': (2, True)}),
        ('proposal', {'person': 2, 'assignment': False, 'outcome': 'ok'}),
        ('proposal', {'person': 4, 'assignment': True, 'outcome': 'ok'}),
        ('proposal',
          {'person': 3,
          'assignment': True,
          'outcome': 'conflict',
          'conflict_statement': (3, True)}),
        ('proposal', {'person': 3, 'assignment': False, 'outcome': 'ok'}),
        ('proposal',
          {'person': 1,
          'assignment': True,
          'outcome': 'conflict',
          'conflict_statement': (1, True)}),
        ('proposal', {'person': 1, 'assignment': False, 'outcome': 'ok'}),
        ('success', {'assignments': (False, False, False, False, True)})
    ]
    tape = generate_chain_of_thoughts(statements, dynamic_person_order=True)
    self.assertEqual(tape, expected_tape)

def test_chain_of_thoughts_format_regression(self):
  # Regression test: NOTE the correct answer is not unique and it can change when the CoT generator code
  # is changed. So the failure of this test does not necessarily mean the code is incorrect. If the code
  # is changed and verified to be correct, this test can be updated with the new target outputs.
  problem = {
      'statements': (('and', ('telling-truth', 2), ('lying', 3)),
                     ('telling-truth', 2),
                     ('<=>', ('lying', 4), ('telling-truth', 4)),
                     ('and', ('lying', 2), ('lying', 4)),
                     ('lying', 2)),
      'solution': (False, False, False, False, True),
      'all_slutions': [(False, False, False, False, True)]
  }
  chain_of_thoughts = generate_chain_of_thoughts(problem['statements'])
  formatted_problem = {'knight_knave': {'Knave': 'Knave',
                    'Knight': 'Knight',
                    'a_knave': 'a knave',
                    'a_knight': 'a knight',
                    'knave': 'knave',
                    'knight': 'knight'},
   'names': ['David', 'Zoey', 'Alexander', 'Aurora', 'Isabella'],
   'quiz': 'A very special island is inhabited only by knights and knaves. '
           'Knights always tell the truth, and knaves always lie. You meet 5 '
           'inhabitants: David, Zoey, Alexander, Aurora, and Isabella. In '
           'David\'s words: "Alexander is a knight and Aurora is a knave". Zoey '
           'expressed that Alexander is a knight. Alexander said, "Isabella is a '
           'knave if and only if Isabella is a knight." Aurora commented, '
           '"Alexander is a knave and Isabella is a knave". According to '
           'Isabella, "Alexander is a knave". So who is a knight and who is a '
           'knave?',
   'solution': (False, False, False, False, True),
   'solution_text': 'David is a knave, Zoey is a knave, Alexander is a knave, '
                    'Aurora is a knave, and Isabella is a knight.'}
  cot_format = format_chain_of_thoughts(problem, formatted_problem, chain_of_thoughts,
                                        repeat_claim_for_assumption=True,
                                        repeat_claim_for_contradiction=True)
  expected_cot = ('Let us think step by step, by considering whether each person is lying and if that leads to contradiction.',
   ['Assume David is a knight. No contradiction is found in their claim that Alexander is a knight and Aurora is a knave.',
    'Alexander cannot be a knight, because this would contradict the claim of their own.',
    'Alexander cannot be a knave, because this would contradict the claim of David.',
    'We have exhausted all possibilities for Alexander, so let us go back and reconsider David.',
    'Assume David is a knave. No contradiction is found in their false claim that Alexander is a knight and Aurora is a knave.',
    'Alexander cannot be a knight, because this would contradict the claim of their own.',
    'Assume Alexander is a knave. No contradiction is found in their false claim that Isabella is a knave if and only if Isabella is a knight.',
    'Assume Isabella is a knight. No contradiction is found in their claim that Alexander is a knave.',
    'Aurora cannot be a knight, because this would contradict the claim of their own.',
    'Assume Aurora is a knave. No contradiction is found in their false claim that Alexander is a knave and Isabella is a knave.',
    'Zoey cannot be a knight, because this would contradict the claim of their own.',
    'Assume Zoey is a knave. No contradiction is found in their false claim that Alexander is a knight.'],
   'This leads to a feasible solution.')
  self.assertEqual(cot_format, expected_cot)

  cot_format = format_chain_of_thoughts(problem, formatted_problem, chain_of_thoughts,
                                        repeat_claim_for_assumption=False,
                                        repeat_claim_for_contradiction=False)
  expected_cot = ('Let us think step by step, by considering whether each person is lying and if that leads to contradiction.',
   ['Assume David is a knight.',
    'Alexander cannot be a knight, because this would contradict the claim of their own.',
    'Alexander cannot be a knave, because this would contradict the claim of David.',
    'We have exhausted all possibilities for Alexander, so let us go back and reconsider David.',
    'Assume David is a knave.',
    'Alexander cannot be a knight, because this would contradict the claim of their own.',
    'Assume Alexander is a knave.',
    'Assume Isabella is a knight.',
    'Aurora cannot be a knight, because this would contradict the claim of their own.',
    'Assume Aurora is a knave.',
    'Zoey cannot be a knight, because this would contradict the claim of their own.',
    'Assume Zoey is a knave.'],
   'This leads to a feasible solution.')
  self.assertEqual(cot_format, expected_cot)



class KnightsAndKnaves_Environment(VerifiableEnvironment):
    prompt_template = \
r"""{statements}

So who is a knight and who is a knave?

Output Format: You must infer the identity of each character. Your final answer must clearly state the identity of each character by following the format example (do **NOT** include the backticks or quotes): `Ella is a knight, Jacob is a knave, Benjamin is a knight, Lucas is a knave, and Samuel is a knight.`"""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 10.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the KnightsAndKnaves_Environment instance.
        """

        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight": rewarding_weight,
        }

    def _generate(self) -> None:
        """
        Keys in parameter:
            N: number of people for K&K problems.
            depth_constraint: the max depth of each person's statement. The depth refer to the level of
                recursion of operators such as 'and', 'or', etc. Increasing the depth would allow
                increasing the difficulty. Though currently the automatic formatting of the problems
                into nautral languages does not support depth more than 2.
            width_constraint: the max width (number of branches in operators such as 'and', 'or') of each
                person's statement.
        """

        """Generate a new instance of the Knights and Knaves problem."""
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        assert "depth_constraint" in self.parameter, "depth_constraint is required in parameter"
        depth_constraint = self.parameter["depth_constraint"]
        assert depth_constraint >= 1, "depth_constraint should be greater than or equal to 1"

        assert "width_constraint" in self.parameter, "width_constraint is required in parameter"
        width_constraint = self.parameter["width_constraint"]
        assert width_constraint >= 1, "width_constraint should be greater than or equal to 1"

        # Generate problem using the KKProblemSampler
        sampler = KKProblemSampler(
            rand_seed=self.seed,
            n_people=N,
            depth_constraint=depth_constraint,
            width_constraint=width_constraint
        )
        
        # Get a valid problem with unique solution
        problems = sampler.sample_valid_problems(
            n_problems=1,
            max_retry=1000,
            skip_no_solution=True,
            skip_multiple_solutions=True
        )
        
        if not problems:
            raise RuntimeError("Failed to generate a valid problem")
            
        problem = problems[0]
        
        # Format the problem into natural language
        formatter = KKProblemFormatter(rand_seed=self.seed, problem=problem)
        formatted = formatter.format_problem()
        
        # Store the problem data
        self.parameter["names"] = formatted["names"]
        self.parameter["quiz"] = formatted["quiz"]
        self.parameter["reference_answer"] = formatted["solution_text"]
        self.parameter["gold_answer"] = self._process(self.parameter["reference_answer"])

    def _prompt_generate(self) -> str:
        """Generate the prompt for the problem instance."""
        return self.prompt_template.format(
            N = self.parameter["N"],
            names = ", ".join(self.parameter["names"][:-1]) + ", and " + self.parameter["names"][-1],
            statements = self.parameter["quiz"].split("So who is")[0].strip(),
        )

    def _process(self, answer: str) -> Optional[Dict[str, str]]:
        """Parses model's answer text into status dictionary.

        Args:
            answer: Text extracted from model's <answer> tags
        Returns:
            Dictionary mapping character names to predicted roles, or None if incomplete
        """
        status_dict = {}

        if not isinstance(answer, str):
            return None
        
        knight_count = answer.lower().count('knight')
        knave_count = answer.lower().count('knave')

        if knight_count + knave_count != self.parameter["N"]:
            return None
        
        for name in self.parameter["names"]:
            pattern = re.compile(rf'{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)', re.IGNORECASE)
            match = pattern.search(answer)

            if match:
                role = match.group(1).lower()
                status_dict[name] = role
                # print(f"  Found: {name} → {role}")
            else:
                # print(f"  [Error] Missing identification for {name}")
                return None

        return status_dict


    def scorer(self, output: str) -> float:
        """Score the model's output."""
        processed_result = self.processor(output)
        
        if processed_result is None:
            return self.rewards["wrong_format"]
            
        # Convert knight/knave to True/False
        model_solution = []
        true_solution = []
        for name in self.parameter["names"]:
            model_solution.append(processed_result[name])
            true_solution.append(self.parameter["gold_answer"][name])
        
        if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta":
            return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(model_solution, true_solution)) / len(model_solution)) ** self.rewards["rewarding_beta"])
        elif self.rewards["rewarding_strategy"] == "gold=answer":
            return self.rewards["rewarding_weight"] * (model_solution == true_solution)
        else:
            raise NotImplementedError(f"Unknown rewarding strategy: {self.rewards['rewarding_strategy']}")
