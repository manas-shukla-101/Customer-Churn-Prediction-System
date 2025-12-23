"""
Data Preprocessing and Processing Module
Handles data loading, cleaning, encoding, and preparation for model training.
Compatible with various datasets and automatically detects suitable columns.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Handles all data preprocessing tasks in a dataset-friendly manner."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numeric_columns = []
        self.categorical_columns = []
        self.target_column = None
        self.df_processed = None
        
    def load_data(self, file_path):
        """Load data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            print(f"✓ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"✗ Error loading file: {e}")
            return None
    
    def detect_target_column(self, df):
        """Auto-detect churn/target column from common naming patterns."""
        possible_names = ['churn', 'target', 'churned', 'churn_flag', 'is_churn', 
                         'customer_churn', 'churn_status', 'attrition']
        
        for col in possible_names:
            if col.lower() in df.columns.str.lower().values:
                self.target_column = df.columns[df.columns.str.lower() == col.lower()][0]
                return self.target_column
        
        # If no match, ask user or return None
        return None
    
    def identify_column_types(self, df):
        """Identify numeric and categorical columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target from feature columns
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)
        
        self.numeric_columns = numeric_cols
        self.categorical_columns = categorical_cols
        
        return numeric_cols, categorical_cols
    
    def handle_missing_values(self, df):
        """Handle missing values intelligently."""
        # Fill numeric columns with median
        for col in self.numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        for col in self.categorical_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        return df
    
    def encode_target(self, df):
        """Encode target variable to binary (0, 1)."""
        if self.target_column is None:
            return df, None
        
        # Handle both Yes/No and other formats
        if df[self.target_column].dtype == 'object':
            unique_vals = df[self.target_column].unique()
            if len(unique_vals) == 2:
                # Map to 0 and 1
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                df[self.target_column] = df[self.target_column].map(mapping)
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features using LabelEncoder."""
        for col in self.categorical_columns:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_features(self, df, fit=True):
        """Scale numeric features using StandardScaler."""
        if len(self.numeric_columns) == 0:
            return df
        
        if fit:
            df[self.numeric_columns] = self.scaler.fit_transform(df[self.numeric_columns])
        else:
            df[self.numeric_columns] = self.scaler.transform(df[self.numeric_columns])
        
        return df
    
    def preprocess(self, df, fit=True):
        """Complete preprocessing pipeline."""
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode target
        df = self.encode_target(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Scale numeric features
        df = self.scale_features(df, fit=fit)
        
        self.df_processed = df
        return df
    
    def prepare_for_training(self, df):
        """Prepare data for model training (separate X and y)."""
        if self.target_column is None:
            raise ValueError("Target column not identified!")
        
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        return X, y
    
    def prepare_for_prediction(self, df):
        """Prepare data for prediction (remove target column if exists)."""
        df = df.copy()
        
        # Remove target column if it exists
        if self.target_column and self.target_column in df.columns:
            df = df.drop(columns=[self.target_column])
        
        # Ensure all object columns are encoded
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in object_cols:
            if col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Scale numeric columns
        cols_to_scale = [col for col in self.numeric_columns if col in df.columns]
        if cols_to_scale:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        return df
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self):
        """Return list of feature column names."""
        if self.df_processed is not None:
            return [col for col in self.df_processed.columns if col != self.target_column]
        return []
    
    def get_statistics(self, df):
        """Get basic statistics about the dataset."""
        stats = {
            'rows': df.shape[0],
            'columns': df.shape[1],
            'numeric_features': len(self.numeric_columns),
            'categorical_features': len(self.categorical_columns),
            'missing_values': df.isnull().sum().sum(),
            'target_distribution': df[self.target_column].value_counts().to_dict() if self.target_column else None
        }
        return stats
