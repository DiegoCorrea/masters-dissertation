#!/usr/bin/env bash
xgettext -d base -o locales/base.pot src/config/language_strings.py
msgmerge --update locales/pt/LC_MESSAGES/base.po locales/base.pot
msgmerge --update locales/en/LC_MESSAGES/base.po locales/base.pot
msgfmt -o locales/pt/LC_MESSAGES/base.mo locales/pt/LC_MESSAGES/base
msgfmt -o locales/en/LC_MESSAGES/base.mo locales/en/LC_MESSAGES/base
