Course Project on Imperfect Information Monte-Carlo Tree Search algorithms.

The goal is to solve French Belote in two settings : 
- A simplified setting with only 12 cards, where all possible states are tried.
- The full 32 cards game, where the algorithms sample possible states instead of performing exhaustive search.

We compare four methods, implemented from scratch : 
- Cheating UCT (UCT with perfect information)
- All Worlds UCT (trying UCT in many possible worlds) and picking the action which is the best in most worlds
- All Worlds UCT (trying UCT in many possible worlds) and picking the action which has the highest expected total reward when averaging over all worlds
- Single-Observer Information Set Monte-Carlo Tree Search : a more efficient method which shares the computational budget of many worlds in a single search tree, where nodes are information sets instead of perfect information states.

To reproduce the evaluation method, run the `Evaluate.py` script, and in the execution part, set to True or False the two sections, depending on if you want to evaluate on the 12 cards game or on the 32 cards game.