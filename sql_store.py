"""
SQL Store for Structured Data
Stores and queries tables extracted from documents
"""

import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional
import re


class SQLStore:
    """Store and query structured table data using SQLite."""
    
    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize SQL store.
        
        Args:
            db_path: SQLite database path (":memory:" for in-memory)
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.tables = {}  # table_name -> metadata
        
        print(f"✓ SQLStore initialized")
        print(f"  Database: {db_path}")
    
    def add_tables_from_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Extract and store tables from document chunks.
        
        Args:
            chunks: List of chunk dictionaries
        """
        # Filter chunks with table data
        table_chunks = [
            chunk for chunk in chunks
            if chunk.get('metadata', {}).get('type') == 'table' or
               isinstance(chunk.get('content'), list)
        ]
        
        if not table_chunks:
            print("  No tables found in chunks")
            return
        
        print(f"\n  Processing {len(table_chunks)} tables...")
        
        for idx, chunk in enumerate(table_chunks):
            try:
                # Get table content
                content = chunk.get('content', chunk.get('text'))
                
                # Convert to DataFrame
                if isinstance(content, list) and content:
                    df = pd.DataFrame(content)
                elif isinstance(content, str):
                    # Try to parse as CSV
                    from io import StringIO
                    df = pd.read_csv(StringIO(content))
                else:
                    continue
                
                if df.empty:
                    continue
                
                # Create table name from metadata
                metadata = chunk.get('metadata', {})
                source = metadata.get('source', 'unknown')
                
                # Clean table name (alphanumeric + underscore only)
                table_name = re.sub(r'[^\w]', '_', source)
                table_name = f"table_{table_name}_{idx}"
                table_name = table_name[:64]  # Limit length
                
                # Clean column names
                df.columns = [
                    re.sub(r'[^\w]', '_', str(col))[:64]
                    for col in df.columns
                ]
                
                # Create SQL table and insert data
                df.to_sql(
                    table_name,
                    self.conn,
                    if_exists='replace',
                    index=False
                )
                
                # Store metadata
                self.tables[table_name] = {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'source': source,
                    'metadata': metadata
                }
                
                print(f"    ✓ {table_name}: {df.shape[0]} rows × {df.shape[1]} cols")
                
            except Exception as e:
                print(f"    ✗ Error processing table {idx}: {e}")
        
        print(f"\n  ✓ Stored {len(self.tables)} tables")
    
    def query_sql(self, query: str) -> Optional[pd.DataFrame]:
        """
        Execute SQL query on stored tables.
        
        Args:
            query: SQL query string
            
        Returns:
            Query results as DataFrame, or None on error
        """
        try:
            result = pd.read_sql_query(query, self.conn)
            return result
        except Exception as e:
            print(f"SQL query error: {e}")
            return None
    
    def list_tables(self) -> List[str]:
        """
        List all stored tables.
        
        Returns:
            List of table names
        """
        return list(self.tables.keys())
    
    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a table.
        
        Args:
            table_name: Name of table
            
        Returns:
            Table metadata dictionary
        """
        return self.tables.get(table_name)
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")


if __name__ == "__main__":
    # Test the SQL store
    print("SQLStore initialized and ready.")
    store = SQLStore()
    print("Ready to store and query structured table data.")
