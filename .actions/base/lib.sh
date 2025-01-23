#!/bin/bash

error() {
	export failure=true
	notify bangbang "@$GITHUB_ACTOR" "Error $*"
	return 1
}

project() {
	# Return the current project
	if [ -n "$GITHUB_REPOSITORY" ]; then
		return "$GITHUB_REPOSITORY"
	else
		return "$(basename "$(pwd)")"
	fi
}

notify() {
	# Initialization

	icon=":$1:"
	shift
	channel="$1"
	shift
	message="$*"

	# Body
	if [ -z "$GITHUB_ACTOR" ]; then
		# Running in local mode
		echo "$message"
	else
		if [ -z "$CD_WEBHOOK" ]; then
			echo "Please set the CD_WEBHOOK environment variable (you have it in pass)"
			exit 1
		fi

		curl -v -X POST -H "Content-Type: application/json" \
			--data "{\"icon_emoji\":\"$icon\", \"channel\":\"$channel\", \"text\":\
            \"[$message]($GIT_HTTP_SERVER/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_NUMBER)\
            \"}" "$CD_WEBHOOK"
	fi
}

bump() {
	echo "---------------------------------"
	echo "---  Bumping program version  ---"
	echo "---------------------------------"

	# Bump the version
	cz --no-raise 21 bump --changelog --no-verify || error creating the bump with commitizen

	# Push changes
	git remote add ssh "git@$GIT_SERVER:$GITHUB_REPOSITORY.git"
	git pull ssh main || error pulling the main branch in the bump job
	git push ssh main || error pushing the main branch in the bump job
	git push ssh --tags || error pushing the tags in the bump job
}

update_actions() {
	echo "------------------------------"
	echo "---  Updating the actions  ---"
	echo "------------------------------"

	git submodule update --recursive --remote
}
