import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import argparse
import json
from collections import Counter
import matplotlib.pyplot as plt
from query_cards import HearthstoneCardQuery
from hearthstone.deckstrings import Deck, write_deckstring
from hearthstone.enums import FormatType

class HearthstoneDeckBuilder:
    """AI-powered Hearthstone deck builder using machine learning techniques."""

    def __init__(self, db_path="hearthstone_cards.db", model_path="card_model.joblib"):
        """Initialize with paths to the database and model files."""
        self.db_path = db_path
        self.model_path = model_path
        self.card_query = HearthstoneCardQuery(db_path).connect()
        self.conn = sqlite3.connect(self.db_path)
        self.df = None
        self.model = None
        self.card_vectors = None
        self.synergy_matrix = None

    def close(self):
        """Close database connections."""
        self.card_query.close()
        if self.conn:
            self.conn.close()
        return self

    def load_cards_to_dataframe(self):
        """Load all collectible cards from the database into a pandas DataFrame."""
        print("Loading card data...")        # Query to get all collectible cards with their attributes
        # Note: flavor, artist, and collectible are excluded from synergy calculations
        query = """
        SELECT c.id, c.dbfid, c.name, c.cost, c.attack, c.health, c.durability, c.armor,
               cc.name as card_class, ct.name as card_type, r.name as rarity,
               cs.name as card_set, c.text, c.targetingArrowText, cs.name
        FROM cards c
        LEFT JOIN card_classes cc ON c.card_class_id = cc.id
        LEFT JOIN card_types ct ON c.card_type_id = ct.id
        LEFT JOIN rarities r ON c.rarity_id = r.id
        LEFT JOIN card_sets cs ON c.card_set_id = cs.id
        WHERE collectible = 1
        AND ct.name IN ('MINION', 'SPELL', 'WEAPON', 'LOCATION', 'ENCHANTMENT')
        """

        self.df = pd.read_sql_query(query, self.conn)

        # Get mechanics for each card
        mechanics_query = """
        SELECT cm.card_id, GROUP_CONCAT(cm.mechanic, ',') as mechanics
        FROM card_mechanics cm
        GROUP BY cm.card_id
        """

        mechanics_df = pd.read_sql_query(mechanics_query, self.conn)

        # Merge mechanics with the main dataframe
        if not mechanics_df.empty:
            self.df = pd.merge(self.df, mechanics_df, left_on='id', right_on='card_id', how='left')
        else:
            self.df['mechanics'] = None

        # Fill NaN values
        self.df.fillna({
            'attack': 0,
            'health': 0,
            'durability': 0,
            'armor': 0,
            'text': '',
            'flavor': '',
            'artist': '',
            'mechanics': '',
            'targetingArrowText': ''
        }, inplace=True)

        print(f"Loaded {len(self.df)} collectible cards.")
        return self

    def _preprocess_cards(self):
        """Preprocess card data for machine learning."""
        print("Preprocessing card data...")

        # Convert all object/string columns to categorical and then one-hot encode them
        # First, identify all string columns
        string_columns = self.df.select_dtypes(include=['object']).columns.tolist()

        # Remove columns we don't want to one-hot encode
        exclude_columns = ['id', 'name', 'flavor', 'mechanics']
        categorical_features = [col for col in string_columns if col not in exclude_columns]

        print(f"One-hot encoding {len(categorical_features)} categorical features")

        # One-hot encode all categorical variables
        df_encoded = pd.get_dummies(self.df, columns=categorical_features)

        # Create feature for each mechanic
        if 'mechanics' in self.df.columns:
            mechanics = set()
            for mech_str in self.df['mechanics'].dropna():
                if mech_str:
                    mechs = mech_str.split(',')
                    mechanics.update(mechs)

            for mech in mechanics:
                df_encoded[f'mechanic_{mech}'] = df_encoded['mechanics'].apply(
                    lambda x: 1 if x and mech in x.split(',') else 0
                )

        # Select only numerical features for ML
        # First, get all column names
        all_columns = df_encoded.columns.tolist()
        # Then exclude non-numerical columns
        exclude_cols = ['id', 'dbfid', 'name', 'flavor', 'artist',
                      'mechanics', 'card_id', 'collectible']
        # Create feature columns list with only columns that exist in the dataframe
        feature_cols = [col for col in all_columns if col not in exclude_cols
                       and col in df_encoded.columns]

        # Ensure all selected columns are numeric
        df_numeric = df_encoded[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        X = df_numeric.values

        print(f"Final feature matrix shape: {X.shape}")

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Store feature names for later
        self.feature_names = feature_cols
        self.X_scaled = X_scaled
        self.card_ids = self.df['id'].values

        return X_scaled

    def train_card_model(self, n_clusters=8, save_model=True):
        """Train a KMeans clustering model on card data to identify card archetypes."""
        if self.df is None:
            self.load_cards_to_dataframe()

        X_scaled = self._preprocess_cards()

        print(f"Training KMeans model with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X_scaled)

        # Add cluster assignments back to dataframe
        self.df['cluster'] = kmeans.labels_

        # Compute card vectors (distance to each cluster center)
        self.card_vectors = kmeans.transform(X_scaled)

        # Calculate card-to-card similarity matrix
        self.synergy_matrix = cosine_similarity(self.card_vectors)

        # Persist the model
        if save_model:
            model_data = {
                'kmeans': kmeans,
                'feature_names': self.feature_names,
                'card_ids': self.card_ids,
                'card_vectors': self.card_vectors,
                'synergy_matrix': self.synergy_matrix
            }
            joblib.dump(model_data, self.model_path)
            print(f"Model saved to {self.model_path}")

        self.model = kmeans
        return self

    def load_model(self):
        """Load a previously trained model."""
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            model_data = joblib.load(self.model_path)
            self.model = model_data['kmeans']
            self.feature_names = model_data['feature_names']
            self.card_ids = model_data['card_ids']
            self.card_vectors = model_data['card_vectors']
            self.synergy_matrix = model_data['synergy_matrix']
            return True
        else:
            print(f"Model file not found at {self.model_path}")
            return False

    def get_card_synergies(self, card_id, top_n=10):
        """Find the top N cards with highest synergy with the given card."""
        if self.df is None:
            self.load_cards_to_dataframe()

        if self.synergy_matrix is None:
            # Try loading the model first
            if not self.load_model():
                # Train a new model if loading fails
                self.train_card_model()

        # Find the index of the card in our data
        card_index = np.where(self.card_ids == card_id)[0]
        if len(card_index) == 0:
            print(f"Card ID {card_id} not found")
            return []

        card_index = card_index[0]

        # Get synergy scores for this card with all other cards
        synergies = self.synergy_matrix[card_index]

        # Get indices of top synergistic cards (excluding the card itself)
        synergies[card_index] = -1  # Exclude the card itself
        top_indices = np.argsort(synergies)[-top_n:][::-1]

        # Get the card IDs and synergy scores
        results = []
        for idx in top_indices:
            synergy_card_id = self.card_ids[idx]
            synergy_score = synergies[idx]
            card = self.card_query.get_card_by_id(synergy_card_id)
            results.append({
                'id': synergy_card_id,
                'name': card['name'],
                'synergy_score': float(synergy_score),
                'card_data': card
            })

        return results

    def generate_deck(self, hero_class, strategy='midrange', format_type='standard', cards_to_include=None, exclude_sets=None):
        """
        Generate a full Hearthstone deck (30 cards) for a specific hero class and strategy.

        Parameters:
        - hero_class: The class to build for (e.g., 'MAGE', 'WARRIOR')
        - strategy: The deck strategy ('aggro', 'midrange', 'control', 'combo')
        - format_type: The deck format ('standard', 'wild', 'classic', etc.)
        - cards_to_include: List of specific card IDs to include in the deck
        - exclude_sets: List of card sets to exclude

        Returns:
        - A deck dictionary with cards and metadata (minimum 30 cards)
        """
        if self.df is None:
            self.load_cards_to_dataframe()

        if self.synergy_matrix is None and not self.load_model():
            self.train_card_model()

        # Initialize deck
        deck_cards = []
        cards_to_include = cards_to_include or []

        # Get format-legal sets
        format_sets = self._get_format_sets(format_type)

        # Filter cards by class (and neutral) and format-legal sets
        valid_classes = [hero_class, 'NEUTRAL']
        class_cards_df = self.df[
            (self.df['card_class'].isin(valid_classes)) &
            (self.df['card_set'].isin(format_sets))
        ]

        print(f"Found {len(class_cards_df)} legal cards for {hero_class} in {format_type} format")

        # Exclude certain sets if specified
        if exclude_sets:
            class_cards_df = class_cards_df[~class_cards_df['card_set'].isin(exclude_sets)]
            print(f"After excluding specified sets, {len(class_cards_df)} cards remain")

        # Add requested cards first (if they're valid for the class and format)
        for card_id in cards_to_include:
            card = self.card_query.get_card_by_id(card_id)
            if card and (card['card_class_name'] == hero_class or card['card_class_name'] == 'NEUTRAL'):
                # Check if card is format-legal
                if card['card_set_name'] in format_sets:
                    deck_cards.append(card)
                else:
                    print(f"Warning: Card {card['name']} ({card_id}) is not legal in {format_type} format")

        # Set up strategy parameters
        strategy_params = {
            'aggro': {
                'cost_distribution': {0: 0.15, 1: 0.25, 2: 0.25, 3: 0.2, 4: 0.1, 5: 0.05, 6: 0.0, 7: 0.0},
                'card_type_weights': {'MINION': 0.8, 'SPELL': 0.2, 'WEAPON': 0.0},
                'minion_priority': 'attack'
            },
            'midrange': {
                'cost_distribution': {0: 0.05, 1: 0.15, 2: 0.2, 3: 0.25, 4: 0.2, 5: 0.1, 6: 0.05, 7: 0.0},
                'card_type_weights': {'MINION': 0.6, 'SPELL': 0.3, 'WEAPON': 0.1},
                'minion_priority': 'balanced'
            },
            'control': {
                'cost_distribution': {0: 0.05, 1: 0.1, 2: 0.15, 3: 0.15, 4: 0.2, 5: 0.15, 6: 0.1, 7: 0.1},
                'card_type_weights': {'MINION': 0.4, 'SPELL': 0.5, 'WEAPON': 0.1},
                'minion_priority': 'health'
            },
            'combo': {
                'cost_distribution': {0: 0.1, 1: 0.15, 2: 0.2, 3: 0.2, 4: 0.15, 5: 0.1, 6: 0.05, 7: 0.05},
                'card_type_weights': {'MINION': 0.5, 'SPELL': 0.45, 'WEAPON': 0.05},
                'minion_priority': 'effect'
            }
        }

        # Use the appropriate strategy parameters
        params = strategy_params.get(strategy.lower(), strategy_params['midrange'])

        # First, add legendary cards (maximum 1 copy per card)
        legendaries = class_cards_df[class_cards_df['rarity'] == 'LEGENDARY']
        # Sort by synergy with already selected cards
        legendaries = self._sort_by_synergy_with_deck(legendaries, deck_cards)
        for _, card_row in legendaries.head(5).iterrows():
            if len(deck_cards) >= 30:
                break
            card = self.card_query.get_card_by_id(card_row['id'])
            if card and card['id'] not in [c['id'] for c in deck_cards]:
                deck_cards.append(card)        # Calculate how many more cards we need
        cards_needed = 30 - len(deck_cards)

        # Check if any special cards in the deck allow for an expanded deck size
        allows_expanded, max_size = self._allows_expanded_deck(deck_cards)

        # Calculate initial card type targets based on strategy
        card_type_targets = {}
        current_type_counts = self._count_card_types_by_name(deck_cards)

        # Initialize targets for each card type based on strategy weights
        for card_type, weight in params['card_type_weights'].items():
            # Calculate how many of this card type we want in the deck
            target_count = int(max_size * weight)
            # How many do we already have
            current_count = current_type_counts.get(card_type, 0)
            # How many more we need to add
            to_add = max(0, target_count - current_count)
            card_type_targets[card_type] = to_add

        print(f"Initial card type targets: {card_type_targets}")

        # First, distribute cards by type and cost to match strategy
        for card_type, type_target in card_type_targets.items():
            if type_target <= 0:
                continue

            print(f"Adding {type_target} {card_type} cards")

            # For this card type, distribute across costs according to mana curve
            for cost, proportion in params['cost_distribution'].items():
                # How many of this cost we want for this type
                cost_target_for_type = max(1, int(type_target * proportion))

                # Get cards of this cost and type
                type_cost_cards = class_cards_df[
                    (class_cards_df['cost'] == cost) &
                    (class_cards_df['card_type'] == card_type)
                ]

                if type_cost_cards.empty:
                    continue

                # Sort by synergy
                type_cost_cards = self._sort_by_synergy_with_deck(type_cost_cards, deck_cards)

                # Add cards of this type and cost
                cards_added = 0
                for _, card_row in type_cost_cards.iterrows():
                    if len(deck_cards) >= max_size or cards_added >= cost_target_for_type:
                        break

                    card = self.card_query.get_card_by_id(card_row['id'])

                    # Skip if we already have two of this card or it's legendary
                    if card and (card['rarity_name'] == 'LEGENDARY' and
                                card['id'] in [c['id'] for c in deck_cards]):
                        continue

                    # Skip if we already have two of this card
                    if card and sum(1 for c in deck_cards if c['id'] == card['id']) >= 2:
                        continue

                    deck_cards.append(card)
                    cards_added += 1

                    # Update our tracking of how many more cards of this type we need
                    card_type_targets[card_type] = max(0, card_type_targets[card_type] - 1)

        # Now fill remaining slots by cost if needed
        cards_needed = max_size - len(deck_cards)
        if cards_needed > 0:
            print(f"Filling remaining {cards_needed} slots by mana curve")

            # Fill the deck based on the mana curve distribution
            for cost, proportion in params['cost_distribution'].items():
                # How many cards we should have at this cost
                target_count = int(max_size * proportion)
                # How many cards we already have at this cost
                current_count = sum(1 for card in deck_cards if card['cost'] == cost)
                # How many more we need
                to_add = target_count - current_count

                if to_add <= 0:
                    continue

                # Get cards of this cost
                cost_cards = class_cards_df[
                    (class_cards_df['cost'] == cost) &
                    ~class_cards_df['id'].isin([c['id'] for c in deck_cards])
                ]

                # Sort by synergy with already selected cards
                cost_cards = self._sort_by_synergy_with_deck(cost_cards, deck_cards)

                # Add cards of this cost
                for _, card_row in cost_cards.iterrows():
                    if len(deck_cards) >= max_size or to_add <= 0:
                        break

                    card = self.card_query.get_card_by_id(card_row['id'])

                    # Skip if we already have two of this card or it's legendary
                    if card and (card['rarity_name'] == 'LEGENDARY' and
                                card['id'] in [c['id'] for c in deck_cards]):
                        continue

                    # Skip if we already have two of this card
                    if card and sum(1 for c in deck_cards if c['id'] == card['id']) >= 2:
                        continue

                    deck_cards.append(card)
                    to_add -= 1
        allows_expanded, max_size = self._allows_expanded_deck(deck_cards)        # Calculate target card type counts based on strategy
        card_type_targets = {}
        current_type_counts = self._count_card_types_by_name(deck_cards)

        for card_type, weight in params['card_type_weights'].items():
            # Calculate how many of this card type we want in the deck
            target_count = int(max_size * weight)
            # How many do we already have
            current_count = current_type_counts.get(card_type, 0)
            # How many more we need to add
            to_add = max(0, target_count - current_count)
            card_type_targets[card_type] = to_add

        print(f"Card type targets: {card_type_targets}")

        # If we still haven't filled the deck, add remaining cards with appropriate type distribution
        remaining_cards = class_cards_df[~class_cards_df['id'].isin([c['id'] for c in deck_cards])]

        # Process each card type according to target distribution
        for card_type, target_count in card_type_targets.items():
            if target_count <= 0:
                continue            # Filter remaining cards by this type
            # The column in the DataFrame is named 'card_type' not 'cardType'
            type_cards = remaining_cards[remaining_cards['card_type'] == card_type]

            if type_cards.empty:
                print(f"No {card_type} cards available for {hero_class}")
                continue

            # Sort by synergy with current deck
            type_cards = self._sort_by_synergy_with_deck(type_cards, deck_cards)

            # Add cards of this type up to the target count
            cards_added = 0
            for _, card_row in type_cards.iterrows():
                if len(deck_cards) >= max_size or cards_added >= target_count:
                    break

                card = self.card_query.get_card_by_id(card_row['id'])

                # Skip if we already have two of this card or it's legendary
                if card and (card['rarity_name'] == 'LEGENDARY' and
                            card['id'] in [c['id'] for c in deck_cards]):
                    continue

                # Skip if we already have two of this card
                if card and sum(1 for c in deck_cards if c['id'] == card['id']) >= 2:
                    continue

                deck_cards.append(card)
                cards_added += 1

                # After adding a card, check if it enables an expanded deck
                if not allows_expanded:  # Only re-check if we weren't already allowed
                    allows_expanded, max_size = self._allows_expanded_deck(deck_cards)        # If we still have space, fill with best synergy cards regardless of type
        if len(deck_cards) < max_size:
            # First try with remaining cards that aren't already at max copies
            remaining_cards = class_cards_df.copy()
            remaining_cards = self._sort_by_synergy_with_deck(remaining_cards, deck_cards)

            for _, card_row in remaining_cards.iterrows():
                if len(deck_cards) >= max_size:
                    break

                card = self.card_query.get_card_by_id(card_row['id'])
                if not card:
                    continue

                # Check duplicate restrictions based on card rarity
                if card['rarity_name'] == 'LEGENDARY':
                    # Can only have one copy of legendary cards
                    if card['id'] in [c['id'] for c in deck_cards]:
                        continue
                else:
                    # Can have up to two copies of non-legendary cards
                    if sum(1 for c in deck_cards if c['id'] == card['id']) >= 2:
                        continue

                deck_cards.append(card)        # Organize the final deck
        final_deck = {
            'class': hero_class,
            'strategy': strategy,
            'format': format_type,
            'card_count': len(deck_cards),
            'cards': deck_cards,
            'mana_curve': self._calculate_mana_curve(deck_cards),
            'card_types': self._count_card_types(deck_cards),
            'deck_code': None
        }

        # Generate the deck code
        deck_code = self.generate_deck_code(final_deck)
        final_deck['deck_code'] = deck_code

        return final_deck

    def _sort_by_synergy_with_deck(self, candidate_cards_df, deck_cards):
        """Sort candidate cards by their synergy with the current deck."""
        # Handle empty cases
        if not deck_cards or candidate_cards_df.empty:
            # If deck is empty or no candidate cards, return cards in original order
            return candidate_cards_df.copy().reset_index(drop=True)

        # Create a new DataFrame with card_id and synergy_score columns
        result_data = []

        # For each candidate card, calculate its synergy with the deck
        for _, row in candidate_cards_df.iterrows():
            card_dict = row.to_dict()  # Keep all original data
            card_id = row['id']

            # Find card in our model
            card_idx_array = np.where(self.card_ids == card_id)[0]

            # Calculate synergy score
            if len(card_idx_array) > 0:
                card_idx = card_idx_array[0]
                synergy_sum = 0
                valid_synergies = 0

                # Sum synergies with all cards in deck
                for deck_card in deck_cards:
                    deck_card_idx_array = np.where(self.card_ids == deck_card['id'])[0]
                    if len(deck_card_idx_array) > 0:
                        synergy_sum += self.synergy_matrix[card_idx, deck_card_idx_array[0]]
                        valid_synergies += 1

                # Calculate average synergy (avoid division by zero)
                card_dict['synergy_score'] = synergy_sum / max(1, valid_synergies)
            else:
                # Card not in model, assign zero synergy
                card_dict['synergy_score'] = 0

            result_data.append(card_dict)

        # Create new DataFrame with all original data plus synergy scores
        if not result_data:
            return candidate_cards_df.copy().reset_index(drop=True)

        result_df = pd.DataFrame(result_data)

        # Sort by synergy score
        result_df = result_df.sort_values('synergy_score', ascending=False)

        # Remove the temporary synergy score column and reset index
        if 'synergy_score' in result_df.columns:
            result_df = result_df.drop('synergy_score', axis=1)

        return result_df.reset_index(drop=True)

    def _calculate_mana_curve(self, cards):
        """Calculate the mana curve of a deck."""
        mana_curve = {}
        for i in range(8):  # 0-7+ mana
            cost = i
            if i == 7:
                # 7+ mana
                count = sum(1 for card in cards if card['cost'] >= 7)
            else:
                count = sum(1 for card in cards if card['cost'] == cost)
            mana_curve[str(cost)] = count
        return mana_curve

    def _count_card_types(self, cards):
        """Count the different card types in a deck."""
        card_types = {}
        for card in cards:
            card_type = card['card_type_name']
            card_types[card_type] = card_types.get(card_type, 0) + 1
        return card_types

    def _count_card_types_by_name(self, deck_cards):
        """Count the occurrence of each card type in the deck by name."""
        card_type_counts = {'MINION': 0, 'SPELL': 0, 'WEAPON': 0, 'HERO': 0, 'LOCATION': 0, 'OTHER': 0}

        for card in deck_cards:
            card_type = card['card_type_name']
            if card_type in card_type_counts:
                card_type_counts[card_type] += 1
            else:
                card_type_counts['OTHER'] += 1

        return card_type_counts

    def _allows_expanded_deck(self, deck_cards):
        """
        Checks if the deck contains any card that allows for more than 30 cards.

        Returns:
            tuple: (bool, int) - Whether expanded deck is allowed, and the new maximum size
        """
        # List of cards that modify deck size rules
        # Format: (card_id, new_max_size)``
        special_cards = [
            ('CORE_REV_018', 40),  # Prince Renathal
            ('REV_0189', 40),  # Prince Renathal
        ]

        # Check if any special cards are in the deck
        for card in deck_cards:
            for special_card_id, max_size in special_cards:
                if card['id'] == special_card_id:
                    return True, max_size

        # No special cards found
        return False, 30

    def _get_format_sets(self, format_type='standard'):
        """
        Get a list of card sets that are legal in the specified format.

        Parameters:
        - format_type: The deck format ('standard', 'wild', 'classic', 'twist')

        Returns:
        - A list of card set names that are legal in the specified format
        """
        # Query the database for format-legal sets using the format flags
        format_type = format_type.lower()

        if format_type == 'wild':
            # Wild format allows all collectible sets
            query = """
            SELECT name FROM card_sets
            WHERE wild = 1
            """
        elif format_type == 'standard':
            # Standard format allows only standard-flagged sets
            query = """
            SELECT name FROM card_sets
            WHERE standard = 1
            """
        elif format_type == 'classic':
            # Classic format allows only classic-flagged sets
            query = """
            SELECT name FROM card_sets
            WHERE classic = 1
            """
        elif format_type == 'twist':
            # Twist format is currently the same as standard, but could be customized later
            # For now, use standard-flagged sets
            query = """
            SELECT name FROM card_sets
            WHERE twist = 1
            """
        else:
            # Default to standard format
            query = """
            SELECT name FROM card_sets
            WHERE standard = 1
            """

        # Execute query
        cursor = self.conn.cursor()
        cursor.execute(query)

        # Get set names
        results = cursor.fetchall()
        format_sets = [row[0] for row in results]

        print(f"Format {format_type} legal sets: {format_sets}")
        return format_sets

    def plot_mana_curve(self, deck):
        """Plot the mana curve of a deck."""
        mana_curve = deck['mana_curve']
        costs = [int(cost) for cost in mana_curve.keys()]
        counts = list(mana_curve.values())

        plt.figure(figsize=(10, 6))
        plt.bar(costs, counts)
        plt.title(f"{deck['class']} {deck['strategy'].title()} Deck - Mana Curve")
        plt.xlabel('Mana Cost')
        plt.ylabel('Number of Cards')
        plt.xticks(costs, [cost if cost < 7 else '7+' for cost in costs])
        plt.grid(axis='y', alpha=0.3)

        # Save the plot as an image
        plt.savefig('deck_mana_curve.png')
        plt.close()

        print("Mana curve plot saved as 'deck_mana_curve.png'")

    def export_deck(self, deck, file_path):
        """Export a deck to a JSON file."""
        # Convert deck cards to serializable format
        serialized_deck = {
            'class': deck['class'],
            'strategy': deck['strategy'],
            'card_count': deck['card_count'],
            'mana_curve': deck['mana_curve'],
            'card_types': deck['card_types'],
            'deck_code': deck['deck_code'],
            'cards': []
        }

        for card in deck['cards']:
            serialized_card = {
                'id': card['id'],
                'name': card['name'],
                'cost': card['cost'],
                'card_class': card['card_class_name'],
                'card_type': card['card_type_name'],
                'rarity': card['rarity_name']
            }
            serialized_deck['cards'].append(serialized_card)

        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serialized_deck, f, indent=4)

        print(f"Deck exported to {file_path}")
    def generate_deck_code(self, deck):
        """
        Generate a Hearthstone deck code that can be imported directly into the game.
        Uses the hearthstone-deckstrings library to create a valid deck code.

        Parameters:
        - deck: A deck dictionary as returned by generate_deck()

        Returns:
        - A deck code string that can be imported into Hearthstone
        """
        # Get the class ID from the class name
        class_query = "SELECT id FROM card_classes WHERE name = ?"
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()

        cursor.execute(class_query, (deck['class'],))
        class_row = cursor.fetchone()
        if not class_row:
            print(f"Error: Class {deck['class']} not found in database")
            return None

        class_id = class_row['id']

        # Map the format string to FormatType enum
        format_mapping = {
            'standard': FormatType.FT_STANDARD,
            'wild': FormatType.FT_WILD,
            'classic': FormatType.FT_CLASSIC,
            'twist': FormatType.FT_TWIST
        }
        format_type = format_mapping.get(deck.get('format', 'standard').lower(), FormatType.FT_STANDARD)

        # Create a new Deck object with the specified format
        hs_deck = Deck()
        hs_deck.heroes = [class_id]
        hs_deck.format = format_type

        # Debug output
        print(f"Class ID: {class_id}, Format: {format_type}")
        print(f"Total cards in deck: {len(deck['cards'])}")        # Count cards by their dbf_id - use a dictionary to track duplicates
        cards_by_dbfid = {}

        # First, count how many times each card ID appears in the deck
        card_id_count = {}
        for card in deck['cards']:
            card_id_count[card['id']] = card_id_count.get(card['id'], 0) + 1

        # Now process each unique card ID
        for card_id, count in card_id_count.items():
            # Get the dbfid for the card
            query = "SELECT dbfid FROM cards WHERE id = ?"
            cursor.execute(query, (card_id,))
            card_row = cursor.fetchone()

            if not card_row:
                print(f"Warning: Card ID {card_id} not found in database")
                continue

            dbfid = card_row['dbfid']
            # Add the count for this card
            cards_by_dbfid[dbfid] = cards_by_dbfid.get(dbfid, 0) + count

            # Print the card info for debugging
            print(f"Card ID {card_id} occurs {count} times, dbfid: {dbfid}")

        # Debug output of card counts
        print(f"Card count by dbfid: {len(cards_by_dbfid)} unique cards")

        # Convert the dictionary to the format required by hearthstone-deckstrings
        deck_cards = []
        for dbfid, count in cards_by_dbfid.items():
            # The hearthstone library expects a list of [dbfid, count] pairs
            deck_cards.append([dbfid, count])
            #print(f"Added card dbfid {dbfid} with count {count}")

        # Generate the deck code
        try:
            # Explicitly set all required parameters for write_deckstring
            deck_code = write_deckstring(
                cards=deck_cards,
                heroes=[class_id],
                format=format_type
            )
            return deck_code
        except Exception as e:
            print(f"Error generating deck code: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='AI-powered Hearthstone deck builder')

    # Define command-line arguments
    parser.add_argument('--train', action='store_true', help='Train a new card model')
    parser.add_argument('--clusters', type=int, default=8, help='Number of clusters for KMeans model')
    parser.add_argument('--build-deck', action='store_true', help='Build a deck')
    parser.add_argument('--class', dest='hero_class', type=str, help='Hero class for the deck')
    parser.add_argument('--strategy', type=str, choices=['aggro', 'midrange', 'control', 'combo'],
                      default='midrange', help='Deck strategy')
    parser.add_argument('--format', dest='format_type', type=str, choices=['standard', 'wild', 'classic', 'twist'],
                      default='standard', help='Deck format (standard, wild, classic, twist)')
    parser.add_argument('--include-cards', type=str, help='Comma-separated list of card IDs to include')
    parser.add_argument('--exclude-sets', type=str, help='Comma-separated list of card sets to exclude')
    parser.add_argument('--card-synergies', type=str, help='Get cards with synergy to this card ID')
    parser.add_argument('--output', type=str, default='deck.json', help='Output file path for deck')

    args = parser.parse_args()

    # Initialize the deck builder
    builder = HearthstoneDeckBuilder()

    try:
        # Train a new model if requested
        if args.train:
            builder.load_cards_to_dataframe()
            builder.train_card_model(n_clusters=args.clusters)

        # Get card synergies if requested
        if args.card_synergies:
            synergies = builder.get_card_synergies(args.card_synergies, top_n=15)
            print(f"Top synergies for card {args.card_synergies}:")
            for i, card in enumerate(synergies, 1):
                print(f"{i}. {card['name']} (Score: {card['synergy_score']:.4f})")

        # Build a deck if requested
        if args.build_deck:
            if not args.hero_class:
                print("Error: You must specify a hero class to build a deck.")
                return

            # Parse card IDs to include
            cards_to_include = []
            if args.include_cards:
                cards_to_include = args.include_cards.split(',')

            # Parse sets to exclude
            exclude_sets = []
            if args.exclude_sets:
                exclude_sets = args.exclude_sets.split(',')

            # Generate the deck
            deck = builder.generate_deck(
                args.hero_class.upper(),
                strategy=args.strategy,
                format_type=args.format_type,
                cards_to_include=cards_to_include,
                exclude_sets=exclude_sets
            )            # Print deck info in Hearthstone format
            print(f"### {deck['class']} {deck['strategy'].title()} Deck")
            print(f"# Class: {deck['class']}")
            print(f"# Format: {deck['format'].title()}")
            print("#")            # Count card occurrences by name only (ignoring ID)
            card_counts_by_name = {}
            for card in deck['cards']:
                card_counts_by_name[card['name']] = card_counts_by_name.get(card['name'], 0) + 1

            # Group cards by mana cost for sorted output
            by_cost = {}
            processed_names = set()  # Track which card names we've already processed

            # First, sort all cards by cost
            for card in deck['cards']:
                cost = card['cost']
                if cost not in by_cost:
                    by_cost[cost] = []

                # Only add each unique card name once to the cost group
                if card['name'] not in processed_names:
                    by_cost[cost].append(card)
                    processed_names.add(card['name'])

            # Print cards in mana cost order
            for cost in sorted(by_cost.keys()):
                cost_cards = by_cost[cost]
                # Sort by name within each cost group
                for card in sorted(cost_cards, key=lambda x: x['name']):
                    count = card_counts_by_name[card['name']]
                    print(f"# {count}x ({cost}) {card['name']}")

            # Print special cards with modules if any exist (like Zilliax)
            # This would require additional logic to identify and group module cards

            # Print extra statistics in a comment section
            #print("\n# Card counts by type:")
            #for card_type, count in deck['card_types'].items():
            #    print(f"# {card_type}: {count}")

            #print("\n# Mana curve:")
            #for cost, count in deck['mana_curve'].items():
            #    cost_label = cost if int(cost) < 7 else "7+"
            #    print(f"# {cost_label}: {count}")

            print("\nDeck Code:")
            print(deck['deck_code'])

            link = f"https://hearthstone.blizzard.com/en-us/deckbuilder?deckcode={deck['deck_code']}&class={deck['class']}&deckFormat={deck['format']}"
            print(f"Link: {link}")

            # Plot the mana curve
            builder.plot_mana_curve(deck)

            # Export the deck
            builder.export_deck(deck, args.output)

    finally:
        # Clean up
        builder.close()


if __name__ == '__main__':
    main()
