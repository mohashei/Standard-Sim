"""
Contains useful functions for dataset classes.
"""

def convert_action_to_int(actions):
	""" Converts a string of an action to an integer
	"""
	action_ints = []
	for action in actions:
		if action == "take":
			action_ints.append(1)
		elif action == "put":
			action_ints.append(2)
		elif action == "shift":
			action_ints.append(3)
		else:
			print("Action ", action, " not recognized!")
			raise Exception
	return action_ints