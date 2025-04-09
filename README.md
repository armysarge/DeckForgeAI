[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/armysarge/DeckForgeAI)

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Donate-brightgreen?logo=buymeacoffee)](https://www.buymeacoffee.com/armysarge)

[![SQLLite](https://img.shields.io/badge/SQLite-3.8%2B-blue.svg)](https://www.sqlite.org/index.html)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-blue.svg)](https://scikit-learn.org/)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/armysarge/DeckForgeAI)](https://github.com/armysarge/DeckForgeAI/issues)

# DeckForgeAI

A powerful toolset for managing Hearthstone decks and querying the complete Hearthstone card database. Includes AI-powered deckbuilding to optimize strategies and streamline gameplay planning.

## Project Roadmap

Current development status as of April 9, 2025:
- ✅ Build SQLite DB (Done)
- ✅ Query cards (Complete)
- 🔄 AI Deck Building (ALPHA - Still being refined)
- 🔄 Deck Management (In progress)

## Features

- Fetches all Hearthstone cards from the HearthstoneJSON API
- Stores cards in a SQLite database with comprehensive card attributes
- Provides a query tool for searching and filtering cards
- **AI-Powered Deck Building (ALPHA):**
  - ⚠️ **Note:** The AI deck creation feature is currently in ALPHA stage. Generated decks may not always make complete sense or be fully optimized for competitive play.
  - Analyzes card synergies using machine learning
  - Creates decks based on different strategies (aggro, midrange, control, combo)
  - Generates mana curve visualizations
  - Exports decks to JSON format
  - Get Deck Codes for easy import into Hearthstone
  - Supports multiple deck formats for flexibility

## Requirements

- Python 3.6+
- Required packages:
  - `requests` - For API calls
  - `sqlite3` (included in Python standard library) - For database operations
  - `pandas` - For data manipulation
  - `scikit-learn` - For machine learning algorithms
  - `numpy` - For numerical computations
  - `matplotlib` - For visualization
  - `joblib` - For model persistence

## Installation

```bash
# Install required packages
pip install -r requirements.txt
```

## Usage

### Building the Database

To build or update the card database:

```bash
python buildDB.py
```

This will:
1. Create a new SQLite database named `hearthstone_cards.db` in the current directory (or update if it already exists)
2. Fetch the latest card data from HearthstoneJSON API
3. Insert all cards into the database

### Querying the Database

The `query_cards.py` script provides various options for querying the database:

```bash
# List all card classes
python query_cards.py --list-classes

# List all card sets
python query_cards.py --list-sets

# List all card types
python query_cards.py --list-types

# List all card rarities
python query_cards.py --list-rarities

# Search for cards by name (case insensitive, partial match)
python query_cards.py --name "dragon"

# Search for cards by class
python query_cards.py --class "MAGE"

# Search for cards by mana cost
python query_cards.py --cost 7

# Search for collectible cards only
python query_cards.py --name "dragon" --collectible

# Combine multiple search criteria
python query_cards.py --class "WARLOCK" --rarity "LEGENDARY" --collectible

# Get a specific card by ID
python query_cards.py --id "EX1_572"

# Increase the number of results (default is 10)
python query_cards.py --name "beast" --limit 50
```

## API Information

This project uses the [HearthstoneJSON API](https://hearthstonejson.com/), which provides data about all Hearthstone cards in JSON format.

## License

This is a personal project for educational purposes.

Hearthstone is a registered trademark of Blizzard Entertainment, Inc. This project is not affiliated with or endorsed by Blizzard Entertainment.
