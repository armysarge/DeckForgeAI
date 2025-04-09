#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sqlite3
import json
import argparse


class HearthstoneCardQuery:
    """Class to query the Hearthstone cards database."""

    def __init__(self, db_path="hearthstone_cards.db"):
        """Initialize with the path to the database file."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect(self):
        """Connect to the database."""
        self.conn = sqlite3.connect(self.db_path)
        # Enable dictionary cursor
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        return self

    def _build_card_query(self, include_joins=True):
        """Build the base query for card selection with joins to lookup tables."""
        query = """
        SELECT c.*,
            cc.name as card_class_name,
            cs.name as card_set_name,
            ct.name as card_type_name,
            f.name as faction_name,
            r.name as rarity_name,
            race.name as race_name,
            ss.name as spell_school_name,
            a.name as artist_name
        FROM cards c
        """

        if include_joins:
            query += """
            LEFT JOIN card_classes cc ON c.card_class_id = cc.id
            LEFT JOIN card_sets cs ON c.card_set_id = cs.id
            LEFT JOIN card_types ct ON c.card_type_id = ct.id
            LEFT JOIN factions f ON c.faction_id = f.id
            LEFT JOIN rarities r ON c.rarity_id = r.id
            LEFT JOIN races race ON c.race_id = race.id
            LEFT JOIN spell_schools ss ON c.spell_school_id = ss.id
            LEFT JOIN artists a ON c.artist_id = a.id
            """

        return query

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
        return self

    def get_card_by_id(self, card_id):
        """Get a card by its ID."""
        query = self._build_card_query()
        query += " WHERE c.id = ?"
        self.cursor.execute(query, (card_id,))
        row = self.cursor.fetchone()
        if row:
            card = dict(row)
            self._get_card_mechanics(card)
            return card
        return None

    def _get_card_mechanics(self, card):
        """Get mechanics for a card from the card_mechanics table."""
        if card and 'id' in card:
            self.cursor.execute("SELECT mechanic FROM card_mechanics WHERE card_id = ?", (card['id'],))
            mechanics = [row[0] for row in self.cursor.fetchall()]
            card['mechanics_list'] = mechanics

    def search_cards(self, name=None, card_class=None, rarity=None, card_set=None,
                    card_type=None, collectible_only=False, cost=None, limit=100):
        """
        Search cards by various criteria.
        Returns a list of matching cards.
        """
        query = self._build_card_query()
        query += " WHERE 1=1"
        params = []

        if name:
            query += " AND c.name LIKE ?"
            params.append(f"%{name}%")

        if card_class:
            query += " AND cc.name LIKE ?"
            params.append(f"%{card_class}%")

        if rarity:
            query += " AND r.name LIKE ?"
            params.append(f"%{rarity}%")

        if card_set:
            query += " AND cs.name LIKE ?"
            params.append(f"%{card_set}%")

        if card_type:
            query += " AND ct.name LIKE ?"
            params.append(f"%{card_type}%")

        if collectible_only:
            query += " AND c.collectible = 1"

        if cost is not None:
            query += " AND c.cost = ?"
            params.append(cost)

        query += " LIMIT ?"
        params.append(limit)

        self.cursor.execute(query, params)
        result = [dict(row) for row in self.cursor.fetchall()]

        # Get mechanics for each card
        for card in result:
            self._get_card_mechanics(card)

        return result

    def get_card_classes(self):
        """Get all unique card classes in the database."""
        self.cursor.execute("SELECT name FROM card_classes ORDER BY name")
        return [row[0] for row in self.cursor.fetchall()]

    def get_card_sets(self):
        """Get all unique card sets in the database."""
        self.cursor.execute("SELECT name FROM card_sets ORDER BY name")
        return [row[0] for row in self.cursor.fetchall()]

    def get_card_types(self):
        """Get all unique card types in the database."""
        self.cursor.execute("SELECT name FROM card_types ORDER BY name")
        return [row[0] for row in self.cursor.fetchall()]

    def get_rarities(self):
        """Get all unique rarities in the database."""
        self.cursor.execute("SELECT name FROM rarities ORDER BY name")
        return [row[0] for row in self.cursor.fetchall()]

    def get_races(self):
        """Get all unique races in the database."""
        self.cursor.execute("SELECT name FROM races ORDER BY name")
        return [row[0] for row in self.cursor.fetchall()]

    def get_spell_schools(self):
        """Get all unique spell schools in the database."""
        self.cursor.execute("SELECT name FROM spell_schools ORDER BY name")
        return [row[0] for row in self.cursor.fetchall()]


def print_card(card):
    """Print a card's details in a readable format."""
    if not card:
        print("No card found.")
        return

    print("\n" + "="*50)
    print(f"Card ID: {card['id']}")
    print(f"Name: {card['name']}")
    print(f"Cost: {card['cost']}")

    if card['attack'] is not None:
        print(f"Attack: {card['attack']}")

    if card['health'] is not None:
        print(f"Health: {card['health']}")

    print(f"Type: {card['type']}")
    print(f"Class: {card['cardClass']}")
    print(f"Set: {card['set']}")
    print(f"Rarity: {card['rarity']}")

    if card['text']:
        print(f"Text: {card['text']}")

    if card['flavor']:
        print(f"Flavor: {card['flavor']}")

    if card['mechanics']:
        mechanics = json.loads(card['mechanics']) if card['mechanics'] else []
        if mechanics:
            print(f"Mechanics: {', '.join(mechanics)}")

    if card['race']:
        print(f"Race: {card['race']}")

    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Query the Hearthstone cards database.')
    parser.add_argument('--name', help='Search cards by name')
    parser.add_argument('--class', dest='card_class', help='Filter by card class')
    parser.add_argument('--set', dest='card_set', help='Filter by card set')
    parser.add_argument('--type', dest='card_type', help='Filter by card type')
    parser.add_argument('--rarity', help='Filter by rarity')
    parser.add_argument('--cost', type=int, help='Filter by mana cost')
    parser.add_argument('--collectible', action='store_true', help='Show only collectible cards')
    parser.add_argument('--list-classes', action='store_true', help='List all card classes')
    parser.add_argument('--list-sets', action='store_true', help='List all card sets')
    parser.add_argument('--list-types', action='store_true', help='List all card types')
    parser.add_argument('--list-rarities', action='store_true', help='List all rarities')
    parser.add_argument('--id', help='Get card by ID')
    parser.add_argument('--limit', type=int, default=10, help='Limit number of results (default: 10)')

    args = parser.parse_args()

    query = HearthstoneCardQuery().connect()

    try:
        if args.id:
            card = query.get_card_by_id(args.id)
            if card:
                print_card(card)
            else:
                print(f"No card found with ID: {args.id}")

        elif args.list_classes:
            print("\nAvailable Card Classes:")
            for card_class in query.get_card_classes():
                print(f"- {card_class}")

        elif args.list_sets:
            print("\nAvailable Card Sets:")
            for card_set in query.get_card_sets():
                print(f"- {card_set}")

        elif args.list_types:
            print("\nAvailable Card Types:")
            for card_type in query.get_card_types():
                print(f"- {card_type}")

        elif args.list_rarities:
            print("\nAvailable Rarities:")
            for rarity in query.get_rarities():
                print(f"- {rarity}")

        else:
            # Search for cards based on criteria
            cards = query.search_cards(
                name=args.name,
                card_class=args.card_class,
                rarity=args.rarity,
                card_set=args.card_set,
                card_type=args.card_type,
                collectible_only=args.collectible,
                cost=args.cost,
                limit=args.limit
            )

            if cards:
                print(f"\nFound {len(cards)} matching cards:")
                for card in cards:
                    print_card(card)
            else:
                print("No cards found matching your criteria.")

    finally:
        query.close()


if __name__ == "__main__":
    main()
