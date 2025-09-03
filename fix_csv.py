import os
from pathlib import Path


def fix_csv_file(csv_path: str) -> None:
	data_path = Path(csv_path)
	if not data_path.exists():
		raise FileNotFoundError(f"File not found: {csv_path}")

	backup_path = data_path.with_suffix(".bak")
	# Create backup once if not present
	if not backup_path.exists():
		backup_path.write_bytes(data_path.read_bytes())

	lines = data_path.read_text(encoding="utf-8").splitlines()
	if not lines:
		return

	header = lines[0]
	# Normalize header to 'texte,label'
	header_out = "texte,label"

	out_lines: list[str] = [header_out]
	for raw in lines[1:]:
		if not raw.strip():
			continue
		# Split on last comma to get label
		try:
			texte, label = raw.rsplit(",", 1)
		except ValueError:
			# If malformed, skip line
			continue
		# Escape inner quotes by doubling them
		texte_escaped = texte.replace('"', '""')
		out_lines.append(f'"{texte_escaped}",{label.strip()}')

	data_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
	fix_csv_file(os.path.join("data", "emails.csv"))
	print("Fixed data/emails.csv (backup saved as emails.csv.bak if not already present)")


