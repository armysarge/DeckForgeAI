#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import sqlite3
import requests
import hashlib
import re
from datetime import datetime

# Define current Standard sets
# Note: Update this list when Standard rotation happens
standard_sets = [
    'CORE',
    'FESTIVAL_OF_LEGENDS',
    'PATH_OF_ARTHAS',
    'RETURN_OF_THE_LICH_KING',
    'TITANS',
    'BATTLE_OF_THE_BANDS',
    'WHIZBANGS_WORKSHOP'
]

# Define Classic sets
classic_sets = ['VANILLA']

#Define Twist sets
twist_sets = [
    'CORE',
    'FESTIVAL_OF_LEGENDS',
    'PATH_OF_ARTHAS',
    'RETURN_OF_THE_LICH_KING',
    'TITANS',
    'BATTLE_OF_THE_BANDS',
    'WHIZBANGS_WORKSHOP'
]

class HearthstoneDBBuilder:
    """Class to build a database of all Hearthstone cards using HearthstoneJSON API."""

    def __init__(self, db_path="hearthstone_cards.db"):
        """Initialize with the path to the database file."""
        self.db_path = db_path
        self.api_base_url = "https://api.hearthstonejson.com/v1"
        self.conn = None
        self.cursor = None
        self.language = "enUS"  # Default language
        self.temp_dir = None
        # Cache for lookup tables
        self.lookup_caches = {
            'card_classes': {},
            'card_sets': {},
            'card_types': {},
            'factions': {},
            'rarities': {},
            'races': {},
            'spell_schools': {},
            'artists': {}
        }

    def connect_to_db(self):
        """Connect to the SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        print(f"Connected to database: {self.db_path}")

    def create_metadata_table(self):
        """Create the metadata table."""
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        ''')

        self.conn.commit()
        print("Metadata table created")

    def create_cards_table(self):
        """Create the main cards table with foreign keys to lookup tables for repetitive values.
        This is a placeholder method - actual table creation happens in create_dynamic_cards_table."""
        # The dynamic table creation will be managed by the update_cards_table_schema method
        # after analyzing the JSON structure
        pass

        # Card Tags/Mechanics mapping table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS card_mechanics (
            card_id TEXT,
            mechanic TEXT,
            PRIMARY KEY (card_id, mechanic),
            FOREIGN KEY (card_id) REFERENCES cards (id)
        )
        ''')

        self.conn.commit()
        print("Cards tables created/verified")

    def create_lookup_tables(self):
        """Create lookup tables for repetitive fields to reduce database size."""

        # Card Classes lookup table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS card_classes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
        ''')

        # Card Sets lookup table with collectible flag
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS card_sets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            standard INTEGER DEFAULT 0,
            wild INTEGER DEFAULT 1,
            classic INTEGER DEFAULT 0,
            twist INTEGER DEFAULT 0
        )
        ''')

        # Card Types lookup table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS card_types (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
        ''')

        # Factions lookup table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS factions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
        ''')

        # Rarities lookup table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS rarities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
        ''')

        # Races lookup table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS races (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
        ''')

        # Spell Schools lookup table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS spell_schools (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
        ''')

        # Artists lookup table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS artists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
        ''')

        self.conn.commit()
        print("Lookup tables created/verified")

    def create_tables(self):
        """Create necessary database tables."""
        # Create metadata table
        self.create_metadata_table()

        # Create the cards table
        self.create_cards_table()

        # Create lookup tables
        self.create_lookup_tables()

        self.conn.commit()
        print("Database tables created/verified")

    def get_latest_version(self):
        """Get the latest available version of the HearthstoneJSON API."""
        try:
            response = requests.get(f"{self.api_base_url}/")
            if response.status_code == 200:
                # Extract the version numbers from the HTML (crude but effective)
                html_content = response.text
                version_matches = re.findall(r'href="/v1/(\d+)/"', html_content)

                if version_matches:
                    # Convert to integers and find the highest version
                    version_numbers = [int(v) for v in version_matches]
                    latest_version = max(version_numbers)
                    print(f"Latest HearthstoneJSON API version: {latest_version}")
                    return str(latest_version)

            # Fallback to "latest" which always points to the newest version
            return "latest"
        except Exception as e:
            print(f"Error getting latest version: {e}")
            return "latest"  # Default fallback

    def download_cards_json(self, version="latest"):
        """Download the cards.json file from HearthstoneJSON API."""
        print(f"Downloading cards.json for version {version}...")

        url = f"{self.api_base_url}/{version}/{self.language}/cards.json"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                cards = response.json()
                print(f"Downloaded {len(cards)} cards")
                return cards
            else:
                print(f"Failed to download cards.json: HTTP {response.status_code}")
                return []
        except Exception as e:
            print(f"Error downloading cards.json: {e}")
            return []

    def process_cards_data(self, cards_data):
        """Process the downloaded cards JSON data and use lookup tables for repetitive values."""
        processed_cards = []

        for card in cards_data:
            # Skip cards without an ID
            if 'id' not in card:
                continue

            # Create a dictionary with our standard fields
            processed_card = {
                'id': card.get('id'),
                'dbfId': card.get('dbfId'),
                'name': card.get('name'),
                'text': card.get('text'),
                'flavor': card.get('flavor'),
                'artist': card.get('artist'),  # We'll keep the original value for lookup during store
                'attack': card.get('attack', 0),
                'health': card.get('health', 0),
                'cost': card.get('cost', 0),
                'durability': card.get('durability', 0),
                'armor': card.get('armor', 0),
                'cardClass': card.get('cardClass'),  # Original value for lookup
                'cardSet': card.get('set'),  # 'set' in the API becomes 'cardSet' in our DB
                'cardType': card.get('type'),  # 'type' in the API becomes 'cardType' in our DB
                'faction': card.get('faction'),
                'rarity': card.get('rarity'),
                'race': card.get('race'),
                'collectible': 1 if card.get('collectible') else 0,
                'elite': 1 if card.get('elite') else 0,
                'targetingArrowText': card.get('targetingArrowText'),
                'spellSchool': card.get('spellSchool')
            }

            # Handle mechanics as JSON
            mechanics = card.get('mechanics', [])
            if mechanics:
                processed_card['mechanics'] = json.dumps(mechanics)

            # Handle referencedTags as JSON
            ref_tags = card.get('referencedTags', [])
            if ref_tags:
                processed_card['referencedTags'] = json.dumps(ref_tags)

            processed_cards.append(processed_card)

        return processed_cards

    def calculate_card_hash(self, card):
        """Calculate a hash value for a card to detect changes."""
        # Create a JSON string of the card data in a deterministic way
        # Sort keys to ensure consistent JSON representation
        card_json = json.dumps(card, sort_keys=True)
        # Calculate hash
        return hashlib.sha256(card_json.encode()).hexdigest()

    def store_mechanics(self, card_id, mechanics_json):
        """Store card mechanics in the mapping table."""
        if not mechanics_json:
            return

        try:
            mechanics = json.loads(mechanics_json)
            for mechanic in mechanics:
                self.cursor.execute(
                    "INSERT OR IGNORE INTO card_mechanics (card_id, mechanic) VALUES (?, ?)",
                    (card_id, mechanic)
                )
        except Exception as e:
            print(f"Error storing mechanics for card {card_id}: {e}")

    def update_mechanics(self, card_id, mechanics_json):
        """Update card mechanics in the mapping table."""
        try:
            # First delete existing mechanics for this card
            self.cursor.execute("DELETE FROM card_mechanics WHERE card_id = ?", (card_id,))

            # Then add the new mechanics
            self.store_mechanics(card_id, mechanics_json)
        except Exception as e:
            print(f"Error updating mechanics for card {card_id}: {e}")

    def store_cards(self, cards):
        """Store the cards in the database with hash-based change detection and lookup table references."""
        inserted_count = 0
        updated_count = 0
        unchanged_count = 0
        timestamp = datetime.now().isoformat()

        # Get existing cards and their hashes
        existing_cards = {}
        self.cursor.execute("SELECT id, hash_value, created_at FROM cards")
        for row in self.cursor.fetchall():
            existing_cards[row[0]] = {'hash': row[1], 'created_at': row[2]}

        for card in cards:
            # Skip cards without an id
            if 'id' not in card:
                continue

            card_id = card['id']

            # Calculate hash for current card data
            current_hash = self.calculate_card_hash(card)

            # Check if card exists and if hash has changed
            is_new = card_id not in existing_cards
            is_changed = not is_new and existing_cards[card_id]['hash'] != current_hash

            # Skip if card exists and hasn't changed
            if not is_new and not is_changed:
                unchanged_count += 1
                continue

            # Get or create lookup IDs for repetitive values
            artist_id = self.get_or_create_lookup_id('artists', card.get('artist'))
            card_class_id = self.get_or_create_lookup_id('card_classes', card.get('cardClass'))
            card_set_id = self.get_or_create_lookup_id('card_sets', card.get('cardSet'))
            card_type_id = self.get_or_create_lookup_id('card_types', card.get('cardType'))
            faction_id = self.get_or_create_lookup_id('factions', card.get('faction'))
            rarity_id = self.get_or_create_lookup_id('rarities', card.get('rarity'))
            race_id = self.get_or_create_lookup_id('races', card.get('race'))
            spell_school_id = self.get_or_create_lookup_id('spell_schools', card.get('spellSchool'))

            # Prepare base card data tuple with lookup table IDs instead of direct values
            card_data = [
                card.get('id'),
                card.get('dbfId'),
                card.get('name'),
                card.get('text'),
                card.get('flavor'),
                artist_id,
                card.get('attack'),
                card.get('health'),
                card.get('cost'),
                card.get('durability'),
                card.get('armor'),
                card_class_id,
                card_set_id,
                card_type_id,
                faction_id,
                rarity_id,
                race_id,
                card.get('mechanics'),
                card.get('referencedTags'),
                card.get('collectible', 0),
                card.get('elite', 0),
                card.get('targetingArrowText'),
                spell_school_id,
                current_hash
            ]
            if is_new:
                # For new cards, add created_at and updated_at timestamps
                card_data.extend([timestamp, timestamp])

                # Get all column names from the cards table
                self.cursor.execute("PRAGMA table_info(cards)")
                all_columns = [row[1] for row in self.cursor.fetchall()]

                # Build a dictionary of column names to values
                column_values = {}

                # For all columns in the table, try to find a corresponding value
                for col in all_columns:
                    # Handle special cases first
                    if col == 'hash_value':
                        column_values[col] = current_hash
                    elif col == 'created_at':
                        column_values[col] = timestamp
                    elif col == 'updated_at':
                        column_values[col] = timestamp
                    # Handle lookup table columns
                    elif col == 'artist_id':
                        column_values[col] = artist_id
                    elif col == 'card_class_id':
                        column_values[col] = card_class_id
                    elif col == 'card_set_id':
                        column_values[col] = card_set_id
                    elif col == 'card_type_id':
                        column_values[col] = card_type_id
                    elif col == 'faction_id':
                        column_values[col] = faction_id
                    elif col == 'rarity_id':
                        column_values[col] = rarity_id
                    elif col == 'race_id':
                        column_values[col] = race_id
                    elif col == 'spell_school_id':
                        column_values[col] = spell_school_id
                    else:
                        # Map from API naming to DB column naming
                        api_names = {
                            'dbfId': 'dbfId',
                            'set': 'cardSet',
                            'type': 'cardType',
                            'cardClass': 'cardClass'
                        }

                        # Check if we have a direct match in the card
                        if col in card:
                            column_values[col] = card.get(col)
                        # Check if we have an API name mapping
                        elif col in api_names and api_names[col] in card:
                            column_values[col] = card.get(api_names[col])
                        # Try to get the value directly from card object, will be NULL if not found
                        else:
                            column_values[col] = card.get(col)

                # Extract the column names and values
                columns = list(column_values.keys())
                values = [column_values[col] for col in columns]
                placeholders = ','.join(['?'] * len(values))
                field_names = ', '.join(columns)

                try:
                    self.cursor.execute(
                        f"INSERT INTO cards ({field_names}) VALUES ({placeholders})",
                        values
                    )
                    inserted_count += 1

                    # Store mechanics in the mapping table
                    self.store_mechanics(card_id, card.get('mechanics'))
                except sqlite3.Error as e:
                    print(f"Error inserting card {card_id}: {e}")

            else:
                # For updates, add the updated_at timestamp
                card_data.append(timestamp)
                # Add the card_id as the last parameter for the WHERE clause
                card_data.append(card_id)

                update_fields = "dbfId=?, name=?, text=?, flavor=?, artist_id=?, attack=?, health=?, " \
                                "cost=?, durability=?, armor=?, card_class_id=?, card_set_id=?, card_type_id=?, " \
                                "faction_id=?, rarity_id=?, race_id=?, mechanics=?, referencedTags=?, collectible=?, elite=?, " \
                                "targetingArrowText=?, spell_school_id=?, hash_value=?, updated_at=?"

                try:
                    self.cursor.execute(
                        f"UPDATE cards SET {update_fields} WHERE id=?",
                        card_data[1:]  # Skip the first item (id) as it's used in the WHERE clause
                    )
                    updated_count += 1

                    # Update mechanics in the mapping table
                    self.update_mechanics(card_id, card.get('mechanics'))
                except sqlite3.Error as e:
                    print(f"Error updating card {card_id}: {e}")

        # Store metadata about the update
        self.cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("last_updated", timestamp)
        )

        self.conn.commit()
        print(f"Cards processed: New: {inserted_count}, Updated: {updated_count}, Unchanged: {unchanged_count}")

        return inserted_count + updated_count

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                print(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                print(f"Error cleaning up temporary directory: {e}")

    def build_database(self):
        """Main method to build the complete database."""
        try:
            # Connect to the database
            self.connect_to_db()

            # Create tables
            self.create_tables()

            # Get the latest version
            version = self.get_latest_version()

            print("\n=== DOWNLOADING CARD DATA ===")
            # Download card data
            cards_data = self.download_cards_json(version)
            if not cards_data:
                raise Exception("Failed to download cards data")

            print("\n=== PROCESSING AND STORING CARDS ===")
            # Process and store cards
            if cards_data:
                # Update schema if needed
                field_types = self.analyze_card_schema(cards_data)
                self.update_cards_table_schema(field_types)

                # Process cards
                processed_cards = self.process_cards_data(cards_data)
                print(f"Processed {len(processed_cards)} cards from JSON")

                # Store cards
                if processed_cards:
                    total_updated = self.store_cards(processed_cards)
                    print(f"Successfully stored {total_updated} cards in database")

                    # Update card set format information
                    print("\n=== UPDATING CARD SET FORMAT INFORMATION ===")
                    self.update_card_set_formats()
                else:
                    print("No cards processed from JSON")
            else:
                print("No card data to process")

            # Close database connection
            self.conn.close()
            print("\nDatabase build completed successfully")

        except Exception as e:
            print(f"Error building database: {e}")
            if self.conn:
                self.conn.close()

    def get_card_count(self):
        """Get the count of cards in the database."""
        if not self.conn:
            self.connect_to_db()

        try:
            self.cursor.execute("SELECT COUNT(*) FROM cards")
            count = self.cursor.fetchone()[0]
            return count
        except Exception as e:
            print(f"Error getting card count: {e}")
            return 0

    def analyze_card_schema(self, cards_data):
        """Analyze the card data to determine the required fields and their types"""
        print("Analyzing card schema from JSON data...")

        # Track all possible fields and their inferred types
        field_types = {
            'id': 'TEXT PRIMARY KEY',  # ID is always a primary key
            'created_at': 'TIMESTAMP',
            'updated_at': 'TIMESTAMP',
            'hash_value': 'TEXT'
        }

        # Define fields that should be normalized into lookup tables
        normalized_fields = {
            'artist': 'artist_id',
            'cardClass': 'card_class_id',
            'set': 'card_set_id',  # 'set' in API becomes card_set_id
            'type': 'card_type_id', # 'type' in API becomes card_type_id
            'faction': 'faction_id',
            'rarity': 'rarity_id',
            'race': 'race_id',
            'spellSchool': 'spell_school_id'
        }

        # Analyze each card to collect all possible fields
        for card in cards_data:
            for key, value in card.items():
                # Skip if we already know about this field
                if key in field_types:
                    continue

                # Check if this is a field that should be normalized
                if key in normalized_fields:
                    # Use INTEGER for foreign key fields
                    normalized_field_name = normalized_fields[key]
                    field_types[normalized_field_name] = 'INTEGER'
                    continue

                # Infer the field type based on value type
                if isinstance(value, int):
                    field_types[key] = 'INTEGER'
                elif isinstance(value, bool):
                    field_types[key] = 'INTEGER'  # SQLite doesn't have boolean, use INTEGER
                elif isinstance(value, float):
                    field_types[key] = 'REAL'
                elif isinstance(value, list) or isinstance(value, dict):
                    field_types[key] = 'TEXT'  # Store complex types as JSON text
                else:
                    field_types[key] = 'TEXT'  # Default to TEXT for everything else

        print(f"Discovered {len(field_types)} fields for cards table")
        return field_types

    def update_cards_table_schema(self, field_types):
        """Create or update the cards table schema based on the analyzed field types"""
        # First check if the table exists
        self.cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='cards'")
        table_exists = self.cursor.fetchone()[0] > 0

        # Define the foreign key constraints
        foreign_keys = [
            "FOREIGN KEY (artist_id) REFERENCES artists(id)",
            "FOREIGN KEY (card_class_id) REFERENCES card_classes(id)",
            "FOREIGN KEY (card_set_id) REFERENCES card_sets(id)",
            "FOREIGN KEY (card_type_id) REFERENCES card_types(id)",
            "FOREIGN KEY (faction_id) REFERENCES factions(id)",
            "FOREIGN KEY (rarity_id) REFERENCES rarities(id)",
            "FOREIGN KEY (race_id) REFERENCES races(id)",
            "FOREIGN KEY (spell_school_id) REFERENCES spell_schools(id)"
        ]

        if not table_exists:
            # Build field definitions
            field_definitions = [f"{field} {data_type}" for field, data_type in field_types.items()]

            # Add foreign key constraints
            all_definitions = field_definitions + foreign_keys

            # Create complete CREATE TABLE statement
            create_table_sql = f"CREATE TABLE cards ({', '.join(all_definitions)})"

            print("Creating cards table with schema:")
            print(create_table_sql)

            self.cursor.execute(create_table_sql)
            self.conn.commit()
            print("Cards table created successfully")
        else:
            # Table exists, check for missing columns
            self.cursor.execute("PRAGMA table_info(cards)")
            existing_columns = {row[1] for row in self.cursor.fetchall()}

            # Find columns that need to be added
            missing_columns = {}
            for field, data_type in field_types.items():
                if field not in existing_columns:
                    missing_columns[field] = data_type

            if missing_columns:
                print(f"Found {len(missing_columns)} new columns to add to cards table")
                for field, data_type in missing_columns.items():
                    # Quote the field name if it's a common SQL reserved keyword
                    sql_keywords = ['set', 'type', 'index', 'key', 'order', 'group', 'table', 'column', 'select', 'where', 'from']
                    if field.lower() in sql_keywords:
                        quoted_field = f'"{field}"'
                    else:
                        quoted_field = field

                    alter_table_sql = f'ALTER TABLE cards ADD COLUMN {quoted_field} {data_type}'
                    try:
                        self.cursor.execute(alter_table_sql)
                    except sqlite3.Error as e:
                        print(f"Error adding column {field}: {e}")

                self.conn.commit()
                print("Schema update completed")
            else:
                print("Cards table schema is up to date")

    def get_or_create_lookup_id(self, table_name, value):
        """Get ID for a value from a lookup table, or create it if not exists.

        Args:
            table_name (str): Name of the lookup table
            value (str): The value to look up or create

        Returns:
            int or None: The ID of the value in the lookup table, None if value is None
        """
        if value is None:
            return None

        # Check cache first
        cache_key = table_name
        if cache_key in self.lookup_caches and value in self.lookup_caches[cache_key]:
            return self.lookup_caches[cache_key][value]

        # Not in cache, query the database
        try:
            self.cursor.execute(f"SELECT id FROM {table_name} WHERE name = ?", (value,))
            result = self.cursor.fetchone()

            if result:
                # Found in database, cache and return
                if cache_key in self.lookup_caches:
                    self.lookup_caches[cache_key][value] = result[0]
                return result[0]
            else:
                # Not found, insert new value
                self.cursor.execute(f"INSERT INTO {table_name} (name) VALUES (?)", (value,))
                new_id = self.cursor.lastrowid

                # Cache the new ID
                if cache_key in self.lookup_caches:
                    self.lookup_caches[cache_key][value] = new_id

                return new_id
        except Exception as e:
            print(f"Error with lookup table {table_name} for value '{value}': {e}")
            return None

    def update_card_set_formats(self):
        global standard_sets, twist_sets, classic_sets
        """Update card sets with format information (Standard, Wild, Classic).

        This marks each card set with flags indicating which format(s) it belongs to.
        """
        print("Updating card set format information...")

        # Update Standard sets
        placeholders = ','.join(['?'] * len(standard_sets))
        self.cursor.execute(f"""
        UPDATE card_sets SET standard = 1
        WHERE name IN ({placeholders})
        """, standard_sets)

        # Update twist sets
        placeholders = ','.join(['?'] * len(twist_sets))
        self.cursor.execute(f"""
        UPDATE card_sets SET twist = 1
        WHERE name IN ({placeholders})
        """, twist_sets)

        # Update Classic sets
        placeholders = ','.join(['?'] * len(classic_sets))
        self.cursor.execute(f"""
        UPDATE card_sets SET classic = 1
        WHERE name IN ({placeholders})
        """, classic_sets)

        self.conn.commit()

        # Print format information
        self.cursor.execute("""
        SELECT name, standard, wild, classic, twist
        FROM card_sets
        ORDER BY name
        """)

        set_info = self.cursor.fetchall()

        for row in set_info:
            name, collectible, standard, wild, classic = row
            print(f"{name:<25} {collectible:<12} {standard:<10} {wild:<10} {classic:<10}")

        print("\nCard set format information updated")

if __name__ == "__main__":
    builder = HearthstoneDBBuilder()
    builder.build_database()

    # Display stats
    builder.connect_to_db()
    card_count = builder.get_card_count()
    builder.conn.close()

    print(f"Database contains {card_count} cards")
    print(f"Database file: {os.path.abspath(builder.db_path)}")