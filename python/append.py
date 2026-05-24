from typing import Any, Iterable


def append(x: Any, values: Any, after: int = None):
	"""Mimic R's append(x, values, after = len(x)).

	Works for lists, tuples and other sequence types (strings supported).
	- If after is None, defaults to len(x).
	- If values is not an iterable (or is a single non-string), it's inserted as a single element.
	- If x is a tuple, result is tuple; if x is a string, result is string; otherwise list.
	"""
	# Normalize sequence
	if x is None:
		x_seq = []
	else:
		x_seq = list(x) if not isinstance(x, str) else list(x)

	n = len(x_seq)
	if after is None:
		after = n

	try:
		after = int(after)
	except Exception:
		raise TypeError("after must be an integer or None")

	# clamp after to [0, n]
	if after < 0:
		after = 0
	if after > n:
		after = n

	# Prepare values to insert
	insert_values = None
	# If values is a string, treat as single element only when x is not a string
	if isinstance(values, str) and not isinstance(x, str):
		insert_values = [values]
	else:
		# If it's iterable (but not bytes), expand; otherwise wrap
		if isinstance(values, Iterable) and not isinstance(values, (bytes, bytearray, str)):
			insert_values = list(values)
		else:
			insert_values = [values]

	new_seq = x_seq[:after] + insert_values + x_seq[after:]

	# Return in same type as input
	if isinstance(x, tuple):
		return tuple(new_seq)
	if isinstance(x, str):
		return "".join(new_seq)
	return new_seq


if __name__ == "__main__":
	# quick examples
	print(append([1, 2, 3], [9, 9], after=1))  # [1,9,9,2,3]
	print(append((1, 2), 3, after=0))  # (3,1,2)
	print(append("abc", "X", after=2))  # abXc
