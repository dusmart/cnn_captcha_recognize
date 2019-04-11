#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Implementation of the heuristic method that extract potential arguments
of a syntax-annotated sentence described in:
Lang & Lapata, 2011 "Unsupervised Semantic Role Induction via Split-Merge Clustering"
"""

from functools import reduce

import options
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)







class SyntacticTreeNode:
    """A node (internal or terminal) of a syntactic tree
    :var word_id: int, the CoNLL word id (starts at 1)
    :var word: string, the word contained by the node
    :var lemma: string, the lemma of the word contained by the node
    :var pos: part-of-speech of the node
    :var deprel: string, function attributed by the parser to this word
    :var father: SyntacticTreeBuilder, the father of this node
    :var children: SyntacticTreeNode list, the children of this node
    :var begin: int, the character position this phrase starts at (root would
                be 0)
    :var end: int, the position this phrase ends at (root would be last
              character)
    :var begin_word: int, the position this *word* begins at
    """

    def __init__(self, word_id, word, lemma, plemma, cpos, pos, namedEntityType, features,
                 head, deprel, phead, pdeprel, begin_word, is_predicate):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)

        # self.logger.debug('SyntacticTreeNode({})'.format(deprel))
        self.word_id = word_id

        self.word = word
        self.lemma = lemma
        self.plemma = plemma
        self.cpos = cpos
        self.pos = pos
        self.namedEntityType = namedEntityType
        self.features = features
        self.head = head
        self.deprel = deprel
        self.phead = phead
        self.pdeprel = pdeprel
        self.is_predicate = is_predicate
        self.begin_word = begin_word

        self.father = None
        self.children = []
        self.position = 0

        self.begin, self.end = None, None

    def __repr__(self):
        children = " ".join([str(t.word_id) for t in self.children])

        if self.father:
            father = self.father.word_id
        else:
            father = ""

        return "({}/{}/{}/{}/{}/{}/{}/{}/{} {})".format(
            self.word_id,
            father,
            children,
            self.pos,
            self.deprel,
            self.begin_word,
            self.position,
            self.begin,
            self.end,
            self.word) #, children)

    def __eq__(self, other):
        if isinstance(other, SyntacticTreeNode):
            return ((self.word_id == other.word_id))
        else:
            return False

    def __ne__(self, other):
        return (not self.__eq__(other))

    def __hash__(self):
        return self.word_id

    def __iter__(self):
        for _, child in enumerate(self.children):
            if child.word_id == self.word_id:
                yield(self)
            for node in child:
                if node in self.fathers():
                    yield self
                else:
                    yield node
        if self.position == len(self.children):
            yield self

    def fathers(self, previous=set()):
        if self.father is not None and self.father not in previous:
            result = previous
            #import ipdb; ipdb.set_trace()
            result.add(self.father)
            return self.father.fathers(result)
        return previous

    def flat(self):
        """Return the tokenized sentence from the parse tree."""
        return " ".join([x.word for x in self])

    def contains(self, arg):
        """Search an exact argument in all subtrees"""
        return (self.flat() == arg or
                any((c.contains(arg) for c in self.children)))

    def closest_match(self, arg):
        """Search the closest match to arg"""
        return self.closest_match_as_node(arg).flat().split()

    def closest_match_as_node(self, arg):
        return self._closest_match_as_node_lcs(arg)[1]

    def _closest_match_as_node_lcs(self, arg):
        import distance
        from distance import lcsubstrings as word_overlap

        current_word_list = self.flat().split()
        wanted_word_list = arg.text.split()

        overlap = word_overlap(tuple(current_word_list),
                               tuple(wanted_word_list))
        if not overlap:
            overlap_words = []
        else:
            overlap_words = list(next(iter(overlap)))

        mean_length = (len(current_word_list) + len(wanted_word_list)) / 2
        score = len(overlap_words) / mean_length

        children_results = [c._closest_match_as_node_lcs(arg)
                            for c in self.children]
        return max([(score, self)] + children_results, key=lambda x: x[0])


class SyntacticTreeBuilder():
    """Wrapper class for the building of a syntactic tree
    :var node_dict: every SyntacticTreeNode available by CoNLL word id
    :var father_ids: every dependency relation: child id -> father id
    :var tree_list: every root node, that is every root subtree
    :var sentence: the "sentence" (words separated by spaces)
    """

    def __init__(self, conll_tree):
        """Extract the data provided
        :param conll_tree: The output of the CoNLL parser
        :type conll_tree: str
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)

        self.node_dict, self.father_ids = {}, {}
        self.tree_list = []

        begin = 0
        linenum = 0
        for l in conll_tree.splitlines():
            try:
                linenum += 1
                """Columns from
                https://github.com/aymara/lima/wiki/LIMA-User-Manual"""
                word_id, form, lemma, plemma, cpos, pos, namedEntityType, features, \
                    head, phead, deprel, pdeprel, is_predicate = l.split("\t")[:13]
            except ValueError:
                self.logger.warn('Wrong number of columns (expected 13) in '
                                 'line {}: "{}"\n'.format(linenum, l))
                continue
            word_id = int(word_id)
            head = int(head) if head != '_' else None
            deprel = deprel if deprel != '_' else 'ROOT'

            self.father_ids[word_id] = head

            self.node_dict[word_id] = SyntacticTreeNode(
                word_id=word_id,
                word=form,
                lemma=lemma,
                plemma=plemma,
                cpos=cpos,
                pos=pos,
                namedEntityType=namedEntityType,
                features=features,
                head=head,
                deprel=deprel,
                phead=phead,
                pdeprel=pdeprel,
                is_predicate=is_predicate,
                begin_word=begin)

            begin += 1 + len(form)

        self.sentence = ' '.join([self.node_dict[w_id].word
                                  for w_id in sorted(self.node_dict.keys())])

        # Record father/child relationship
        for word_id, father_id in self.father_ids.items():
            if father_id is not None and father_id != 0:
                try:
                    self.node_dict[word_id].father = self.node_dict[father_id]
                    self.node_dict[father_id].children.append(self.node_dict[word_id])  # noqa
                except KeyError:
                    self.logger.error('father id {} and/or word_id {} not found in CoNLL tree {}'
                                      ''.format(
                                          father_id,
                                          word_id,
                                          conll_tree))

        # Record position: where is father among child?
        # Important to flatten tree
        for father in self.node_dict.values():
            father.position = 0
            for child_id, child in enumerate(father.children):
                if child.begin_word > father.begin_word:
                    father.position = child_id
                    break
                father.position = len(father.children)

        for node in self.node_dict.values():
            if node.father is None:
                # Fill begin/end info
                self.fill_begin_end(node)
                # Fill forest of tree
                self.tree_list.append(node)

    def fill_begin_end(self, node):
        """Fill begin/end values of very subtree"""
        begin_words = [node.begin_word]
        end_words = [node.begin_word + len(node.word) - 1]
        for child in node.children:
            self.fill_begin_end(child)
            begin_words.append(child.begin)
            end_words.append(child.end)
        node.begin = min(begin_words)
        node.end = max(end_words)




















class InvalidRelationError(Exception):
    """Exception raised when trying to build a relation with invalid keywords

    :var error_type: str -- the nature of the wrong keyword
    :var invalid_data: str -- the wrong keyword

    """

    def __init__(self, error_type, invalid_data):
        self.error_type = error_type
        self.invalid_data = invalid_data

    def __str__(self):
        return "Invalid {} : \"{}\"".format(self.error_type, self.invalid_data)


class Relation:
    """A syntactic relation between one node and its neighbour that is closer
    to the predicate. This can be an upward dependency if the node is the
    governor of the relation or a downward dependency in the other case.

    :var name: str -- The name of the relation
    :var direction: str -- Indicates whether this is an upward or downward dependency

    """

    possible_names = {
        "NMOD","P","PMOD","SBJ","OBJ","ROOT","ADV","NAME","VC","COORD","DEP","TMP","CONJ","LOC","AMOD","PRD","APPO","IM","HYPH","HMOD","SUB","OPRD","SUFFIX","TITLE","DIR","POSTHON","MNR","PRP","PRT","LGS","EXT","PRN","LOC-PRD","EXTR","DTV","PUT","GAP-SBJ","GAP-OBJ","DEP-GAP","GAP-PRD","GAP-TMP","PRD-TMP","GAP-LGS","PRD-PRP","BNF","GAP-LOC","DIR-GAP","LOC-OPRD","VOC","GAP-PMOD","EXT-GAP","ADV-GAP","GAP-VC","GAP-NMOD","DTV-GAP","GAP-LOC-PRD","AMOD-GAP","GAP-PRP","EXTR-GAP","DIR-PRD","GAP-MNR","LOC-TMP","MNR-PRD","GAP-PUT","MNR-TMP","DIR-OPRD","LOC-MNR","GAP-OPRD","GAP-SUB"}

    possible_directions = {"UP", "DOWN"}

    def __init__(self, name, direction):
        #logger.debug( "Relation '{}'".format(name))
        if direction not in Relation.possible_directions:
            raise InvalidRelationError("direction", direction)
        if name not in Relation.possible_names:
            raise InvalidRelationError("name", name)

        self.name = name
        self.direction = direction

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
            self.name == other.name and
            self.direction == other.direction)

    def __repr__(self):
        return "{}_{}".format(self.name, self.direction)

    @staticmethod
    def both(name):
        return [Relation(name, "UP"), Relation(name, "DOWN")]


class RelationTreeNode:
    """A node from a tree representation of a sentence, similar to
    a SyntacticTreeNode, except that the root is a given predicate, and that
    all nodes are annotated nwith their relation to their neighbour that is
    closer to the predicate.

    :var node: SyntacticTreeNode -- The matching SyntacticTreeNode
    :var children: RelationTreeNode List -- The children of the node in this
        particular representation
    :var relation: Relation -- The relation between the node and its neighbour
        that is closer to the predicate
    :var status: str -- UNKNOWN, KEPT or DISCARDED depending on whether the
        node is an argument

    """

    def __init__(self, node, children, relation):
        self.node = node
        self.children = children
        self.relation = relation
        self.status = "UNKNOWN"

    def __iter__(self):
        """ Iterate over all the subnodes (node itself is NOT included).
        No particular order is guaranteed.

        """

        for node in self._iter_all():
            if node is not self:
                yield node

    def _iter_all(self):
        for child in self.children:
            for node in child._iter_all():
                yield node
        yield self

    def __repr__(self):
        return "{} {} ({})".format(
            self.node.word, self.relation, ", ".join([str(x) for x in self.children]))

    def keep(self):
        if self.status == "UNKNOWN":
            self.status = "KEPT"

    def discard(self):
        if self.status == "UNKNOWN":
            self.status = "DISCARDED"


def build_relation_tree(node):
    """Transforms a SyntacticTreeNode into a RelationTreeNode

    :param node: The node which will be the root of the resulting tree
    :type node: SyntacticTreeNode
    :returns: RelationTreeNode -- The tree of relation corresponding to the node

    """

    # The root node gets the special relation "IDENTITY", but its relation
    # attribute is never actually read, except for debug purposes.
    return build_relation_tree_rec(node, node, "IDENTITY")


def build_relation_tree_rec(node, new_father, relation):
    """Recursivly build the subtree which starts at node

    :param node: The node from which we start to build the subtree
    :type node: SyntacticTreeNode
    :param new_father: The node that will be the father of this node in the new tree
    :type new_father: SyntacticTreeNode
    :param relation: The relation from ``new_father`` to ``node``
    :type relation: Relation

    """

    # Starts by the syntactic children (except new_father if it is one)
    new_children = [build_relation_tree_rec(x, node, Relation(x.deprel, "DOWN"))
            for x in node.children if x is not new_father]

    # The add the father if it is not new_father
    if not (new_father is node.father or node.father is None):
        new_children.append(build_relation_tree_rec(
            node.father, node, Relation(node.deprel, "UP")))

    return RelationTreeNode(node, new_children, relation)


# Determiner, coordinating conjunction and punctuation POS
# TO is missing because this is also the POS attributed to "to" when it is
# a preposition.
# The list of poncutation POS was obtained by extracting punctuation from
# the fulltext corpus and might therefore be incomplete
rule1_pos = ["CC", "DT", "``", "$", ")", "(", ",", ".", "''", ":"]

# Next two variables were set according to the appendix of the article
rule2_relations = (
    Relation.both("IM") + Relation.both("COORD") + Relation.both("P") + Relation.both("DEP") +
    Relation.both("SUB") + [Relation("PRT", "DOWN"), Relation("OBJ", "UP"),
    Relation("PMOD", "UP"), Relation("ADV", "UP"), Relation("ROOT", "UP"),
    Relation("TMP", "UP"), Relation("SBJ", "UP"), Relation("OPRD", "UP") ])

rule4_relations = reduce(lambda x, y: x + Relation.both(y), [
    "ADV", "AMOD", "APPO", "BNF", "CONJ", "COORD", "DIR", "DTV", "EXT", "EXTR",
    "HMOD", "GAP-OBJ", "LGS", "LOC", "MNR", "NMOD", "OBJ", "OPRD", "POSTHON",
    "PRD", "PRN", "PRP", "PRT", "PUT", "SBJ", "SUB", "SUFFIX", "DEP"
    ], [])


def find_args(predicate_node):
    """ Apply the heuristic to its argument

    :param predicate_node: The node from which we want potential arguments
    :type predicate_node: SyntacticTreeNode

    """

    # Build the relation tree
    tree = build_relation_tree(predicate_node)

    # Apply the 8 rules
    rule1(tree)
    rule2(tree)
    rule3(tree)
    rule4(tree)
    rule5(tree)
    rule6(tree)
    rule7(tree)
    rule8(tree)

    # At this point, all nodes are marked as "KEPT" or "DISCARDED"
    # Returns every node marked as "KEPT"

    # But first, discard nodes which have children which are also candidate arguments
    arg_list = [x for x in tree if x.status == "KEPT"]
    for arg_node in arg_list:
        for subnode in arg_node:
            if subnode in arg_list:
                arg_node.status = "DISCARDED"
                break

    return [x.node for x in tree if x.status == "KEPT"]


def rule1(tree):
    for elem in tree:
        if elem.node.pos in rule1_pos:
            elem.discard()


def rule2(tree):
    for elem in tree:
        if elem.relation in rule2_relations:
            elem.discard()


def rule3(tree):
    candidate = None
    best_position = -1

    for elem in tree:
        if (elem.node.deprel == "SBJ" and
                elem.node.begin_word < tree.node.begin_word and
                elem.node.begin_word > best_position):
            candidate, best_position = elem, elem.node.begin

    if candidate is None:
        return

    found = False
    node = tree.node
    while node is not None:
        if node.begin_word == candidate.node.father.begin_word:
            found = True
            break
        node = node.father

    if found:
        candidate.keep()


def rule4(tree):
    for elem1 in tree.children:
        if elem1.relation in rule4_relations:
            for elem2 in elem1:
                elem2.discard()
        else:
            rule4(elem1)


def rule5(tree):
    for elem in tree:
        if any([x.deprel == "VC" for x in elem.node.children]):
            elem.discard()


def rule6(tree):
    for elem in tree.children:
        # Do not keep elem that are on the left of the predicate
        if (elem.node is not tree.node.father and
        elem.node.begin_word > tree.node.begin_word):
            elem.keep()


def rule7(tree, root=True):
    for elem in tree.children:
        if elem.relation.name == "VC":
            rule7(elem, False)
        elif not root:
            elem.keep()


def rule8(tree):
    for elem in tree:
        elem.discard()








def main():
    parsed_sentence = ("""1	Hoare	hoare	hoare	NNP	NNP	_	_	2	2	NAME	NAME	_	_	_	_
2	Govett	govett	govett	NNP	NNP	_	_	3	3	SBJ	SBJ	_	_	A0	_
3	is	be	be	VBZ	VBZ	_	_	0	0	ROOT	ROOT	_	_	_	_
4	acting	act	act	VBG	VBG	_	_	3	3	VC	VC	Y	act.01	_	_
5	as	as	as	IN	IN	_	_	4	4	ADV	ADV	_	_	A1	_
6	the	the	the	DT	DT	_	_	7	7	NMOD	NMOD	_	_	_	_
7	consortium	consortium	consortium	NN	NN	_	_	10	10	NMOD	NMOD	_	_	_	A2
8	's	's	's	POS	POS	_	_	7	7	SUFFIX	SUFFIX	_	_	_	_
9	investment	investment	investment	NN	NN	_	_	10	10	NMOD	NMOD	_	_	_	_
10	bankers	banker	banker	NNS	NNS	_	_	5	5	PMOD	PMOD	Y	banker.01	_	A0
11	.	.	.	.	.	_	_	3	3	P	P	_	_	_	_
""")
    
    treeBuilder = SyntacticTreeBuilder(parsed_sentence)
    initial_tree_list = treeBuilder.tree_list
  
    #relation_tree = build_relation_tree(initial_tree_list[0].children[0])    
    #print(str(relation_tree))

    #expected = set(["contribution", "more"])
    found = find_args(initial_tree_list[0].children[1])
    print(found)
    #assert(set([x.word for x in found]) == expected)



if __name__ == "__main__":
    main()