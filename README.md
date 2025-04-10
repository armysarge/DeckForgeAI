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

## Querying the Database

The `query_cards.py` script provides various options for querying the database:

```bash
# Query all cards in the database
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
### AI Deck Building (ALPHA)

DeckForgeAI includes machine learning capabilities to analyze card synergies and build decks. Here's how to use these features:

#### 1. Training the AI Model

Before generating decks, you need to train the AI model on your card database:

```bash
# Train a new model with default settings
python ai_deckbuilder.py --train

# Train with a custom number of clusters (advanced)
python ai_deckbuilder.py --train --clusters 12
```

This will:
- Load collectible cards from your database
- Analyze card attributes and synergies using machine learning
- Save the trained model to `card_model.joblib`

Training only needs to be done once, or when you update your card database.

#### 2. Generating Decks

Once the model is trained, you can generate decks with various strategies and formats:

```bash
# Generate a midrange Mage deck (standard format by default)
python ai_deckbuilder.py --build-deck --class MAGE --strategy midrange

# Generate an aggro Hunter deck in wild format
python ai_deckbuilder.py --build-deck --class HUNTER --strategy aggro --format wild

# Generate a control Priest deck in classic format and save to custom file
python ai_deckbuilder.py --build-deck --class PRIEST --strategy control --format classic --output priest_control.json

# Generate a combo Warlock deck in twist format
python ai_deckbuilder.py --build-deck --class WARLOCK --strategy combo --format twist

# Generate a deck in a specific format (default is standard)
python ai_deckbuilder.py --build-deck --class DRUID --strategy midrange --format wild
```

> **Note on Format Rotation:** When Hearthstone Standard format rotates, you'll need to update the sets lists in the `buildDB.py` script to reflect the current legal sets. This ensures that format-based deck generation continues to work correctly, and then recreate the database and re-train the AI model.

Available strategies:
- `aggro`: Fast, aggressive decks with low mana curve
- `midrange`: Balanced decks with efficient minions
- `control`: Defensive decks with removal and late-game value
- `combo`: Decks built around specific card combinations

Available formats:
- `standard`: Current Standard rotation (default)
- `wild`: All cards from all sets
- `classic`: Only cards from the Classic format
- `twist`: Cards eligible for the Twist format

#### 3. Customizing Generated Decks

You can further customize the decks the AI generates:

```bash
# Include specific cards (by card ID)
python ai_deckbuilder.py --build-deck --class WARRIOR --include-cards CORE_EX1_606,EX1_391

# Exclude certain card sets
python ai_deckbuilder.py --build-deck --class MAGE --exclude-sets CORE,LEGACY
```

#### 4. Finding Card Synergies

You can also use the AI to find cards that synergize well with a specific card:

```bash
# Find top 15 cards with synergy to a specific card
python ai_deckbuilder.py --card-synergies EX1_559
```

#### 5. Deck Output

When generating a deck, DeckForgeAI will:
- Display the deck in the console
- Save a mana curve visualization as `deck_mana_curve.png`
- Export the deck to a JSON file (default: `deck.json`)
- Generate a Hearthstone-compatible deck code for easy import into the game

## API Information

This project uses the [HearthstoneJSON API](https://hearthstonejson.com/), which provides data about all Hearthstone cards in JSON format.

## License

This is a personal project for educational purposes.

Hearthstone is a registered trademark of Blizzard Entertainment, Inc. This project is not affiliated with or endorsed by Blizzard Entertainment.
