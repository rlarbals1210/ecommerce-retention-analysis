"""Build a portable Tableau Public workbook from the template and public CSVs.

Requires the optional official tableauhyperapi package.
"""
from pathlib import Path
import tempfile
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
from tableauhyperapi import Connection, CreateMode, HyperProcess, Inserter, SqlType, TableDefinition, TableName, Telemetry
import config

TABLEAU_DIR = config.REPORTS_DIR / 'tableau'
WORKBOOK_NAME = 'Ecommerce Retention and Funnel Analysis'


def package():
    root = ET.parse(TABLEAU_DIR / f'{WORKBOOK_NAME}.twb').getroot()
    types = {'string': SqlType.text(), 'integer': SqlType.big_int(), 'real': SqlType.double(), 'boolean': SqlType.bool()}
    with tempfile.TemporaryDirectory(prefix='retention-tableau-') as tmp:
        tmp = Path(tmp)
        (tmp / 'Data').mkdir()
        with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU, parameters={'log_dir': str(tmp)}) as hyper:
            for source in root.findall('./datasources/datasource'):
                name = source.get('caption')
                frame = pd.read_csv(TABLEAU_DIR / f'{name}.csv')
                columns = source.find('./connection/relation/columns')
                expected = [c.get('name') for c in columns]
                if list(frame.columns) != expected:
                    raise ValueError(f'CSV/template schema mismatch: {name}')
                table = TableDefinition(TableName('Extract', 'Extract'), [TableDefinition.Column(c.get('name'), types[c.get('datatype')]) for c in columns])
                data_path = f'Data/{name}.hyper'
                with Connection(hyper.endpoint, tmp / data_path, CreateMode.CREATE_AND_REPLACE) as conn:
                    conn.catalog.create_schema('Extract')
                    conn.catalog.create_table(table)
                    with Inserter(conn, table) as inserter:
                        inserter.add_rows(frame.itertuples(index=False, name=None))
                        inserter.execute()
                    assert conn.execute_scalar_query('SELECT count(*) FROM "Extract"."Extract"') == len(frame)
                source.find('./extract/connection').set('dbname', data_path)
                for connection in source.findall('./connection/named-connections/named-connection/connection'):
                    connection.set('directory', 'Data')
                (tmp / 'Data' / f'{name}.csv').write_bytes((TABLEAU_DIR / f'{name}.csv').read_bytes())
                print(f'{name}: {len(frame)} rows, {len(expected)} columns verified')
        workbook = tmp / f'{WORKBOOK_NAME}.twb'
        ET.ElementTree(root).write(workbook, encoding='utf-8', xml_declaration=True)
        target = TABLEAU_DIR / f'{WORKBOOK_NAME}.twbx'
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as package_file:
            package_file.write(workbook, workbook.name)
            for artifact in (tmp / 'Data').iterdir():
                package_file.write(artifact, f'Data/{artifact.name}')
    return target


if __name__ == '__main__':
    print(package())
