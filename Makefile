#
# Config:

module = classifier

# --- --- ---

#
# install
install:
	@python3 -m pip install -r requirements.txt

#
# run
run:
	@python3 -m $(module) -C config.json