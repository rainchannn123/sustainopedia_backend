import os
from bw2data import projects, databases, Method, methods
from bw2io import create_default_biosphere3

if 'Tiangong' not in projects:
    projects.create_project('Tiangong')
projects.set_current('Tiangong')

if 'biosphere3' not in databases:
    create_default_biosphere3()

if 'Tiangong' not in databases:
    from bw2data import Database
    db = Database('Tiangong')
    db.register()
    data = {
        ('Tiangong', 'iron_ore_mining'): {
            'name': 'Iron ore mining',
            'unit': 'ton',
            'location': 'CN',
            'type': 'process',
            'exchanges': [
                {
                    'amount': 1.0,
                    'input': ('Tiangong', 'iron_ore_mining'),
                    'type': 'production',
                    'uncertainty type': 0,
                    'loc': 1.0,
                    'scale': 0.0,
                    'shape': 0.0,
                    'minimum': 0.0,
                    'maximum': 0.0
                },
                {
                    'amount': 0.1,
                    'input': ('biosphere3', 'c1b91234-6f24-417b-8309-46111d09c457'),  # NOx
                    'type': 'biosphere',
                    'uncertainty type': 0,
                    'loc': 1.0,
                    'scale': 0.0,
                    'shape': 0.0,
                    'minimum': 0.0,
                    'maximum': 0.0
                }
            ]
        },
        ('Tiangong', 'coal_mining'): {
            'name': 'Coal mining',
            'unit': 'ton',
            'location': 'CN',
            'type': 'process',
            'exchanges': [
                {
                    'amount': 1.0,
                    'input': ('Tiangong', 'coal_mining'),
                    'type': 'production',
                    'uncertainty type': 0,
                    'loc': 1.0,
                    'scale': 0.0,
                    'shape': 0.0,
                    'minimum': 0.0,
                    'maximum': 0.0
                },
                {
                    'amount': 0.2,
                    'input': ('biosphere3', 'c1b91234-6f24-417b-8309-46111d09c457'),  # NOx
                    'type': 'biosphere',
                    'uncertainty type': 0,
                    'loc': 1.0,
                    'scale': 0.0,
                    'shape': 0.0,
                    'minimum': 0.0,
                    'maximum': 0.0
                }
            ]
        },
        ('Tiangong', 'iron_ore_processing'): {
            'name': 'Iron ore processing',
            'unit': 'ton',
            'location': 'CN',
            'type': 'process',
            'exchanges': [
                {
                    'amount': 1.0,
                    'input': ('Tiangong', 'iron_ore_processing'),
                    'type': 'production',
                    'uncertainty type': 0,
                    'loc': 1.0,
                    'scale': 0.0,
                    'shape': 0.0,
                    'minimum': 0.0,
                    'maximum': 0.0
                },
                {
                    'amount': 0.3,
                    'input': ('biosphere3', 'c1b91234-6f24-417b-8309-46111d09c457'),  # NOx
                    'type': 'biosphere',
                    'uncertainty type': 0,
                    'loc': 1.0,
                    'scale': 0.0,
                    'shape': 0.0,
                    'minimum': 0.0,
                    'maximum': 0.0
                }
            ]
        },
        ('Tiangong', 'steelmaking'): {
            'name': 'Steelmaking',
            'unit': 'ton',
            'location': 'CN',
            'type': 'process',
            'exchanges': [
                {
                    'amount': 1.0,
                    'input': ('Tiangong', 'steelmaking'),
                    'type': 'production',
                    'uncertainty type': 0,
                    'loc': 1.0,
                    'scale': 0.0,
                    'shape': 0.0,
                    'minimum': 0.0,
                    'maximum': 0.0
                },
                {
                    'amount': 0.5,
                    'input': ('biosphere3', 'c1b91234-6f24-417b-8309-46111d09c457'),  # NOx
                    'type': 'biosphere',
                    'uncertainty type': 0,
                    'loc': 1.0,
                    'scale': 0.0,
                    'shape': 0.0,
                    'minimum': 0.0,
                    'maximum': 0.0
                }
            ]
        },
        ('Tiangong', 'coking'): {
            'name': 'Coking',
            'unit': 'ton',
            'location': 'CN',
            'type': 'process',
            'exchanges': [
                {
                    'amount': 1.0,
                    'input': ('Tiangong', 'coking'),
                    'type': 'production',
                    'uncertainty type': 0,
                    'loc': 1.0,
                    'scale': 0.0,
                    'shape': 0.0,
                    'minimum': 0.0,
                    'maximum': 0.0
                },
                {
                    'amount': 0.4,
                    'input': ('biosphere3', 'c1b91234-6f24-417b-8309-46111d09c457'),  # NOx
                    'type': 'biosphere',
                    'uncertainty type': 0,
                    'loc': 1.0,
                    'scale': 0.0,
                    'shape': 0.0,
                    'minimum': 0.0,
                    'maximum': 0.0
                }
            ]
        },
        ('Tiangong', 'transportation'): {
            'name': 'Transportation',
            'unit': 'ton-kilometer',
            'location': 'CN',
            'type': 'process',
            'exchanges': [
                {
                    'amount': 1.0,
                    'input': ('Tiangong', 'transportation'),
                    'type': 'production',
                    'uncertainty type': 0,
                    'loc': 1.0,
                    'scale': 0.0,
                    'shape': 0.0,
                    'minimum': 0.0,
                    'maximum': 0.0
                },
                {
                    'amount': 0.1,
                    'input': ('biosphere3', 'c1b91234-6f24-417b-8309-46111d09c457'),  # NOx
                    'type': 'biosphere',
                    'uncertainty type': 0,
                    'loc': 1.0,
                    'scale': 0.0,
                    'shape': 0.0,
                    'minimum': 0.0,
                    'maximum': 0.0
                }
            ]
        },
        ('Tiangong', 'wastewater_discharge'): {
            'name': 'Wastewater discharge',
            'unit': 'ton',
            'location': 'CN',
            'type': 'process',
            'exchanges': [
                {
                    'amount': 1.0,
                    'input': ('Tiangong', 'wastewater_discharge'),
                    'type': 'production',
                    'uncertainty type': 0,
                    'loc': 1.0,
                    'scale': 0.0,
                    'shape': 0.0,
                    'minimum': 0.0,
                    'maximum': 0.0
                },
                {
                    'amount': 0.05,
                    'input': ('biosphere3', 'c1b91234-6f24-417b-8309-46111d09c457'),  # NOx
                    'type': 'biosphere',
                    'uncertainty type': 0,
                    'loc': 1.0,
                    'scale': 0.0,
                    'shape': 0.0,
                    'minimum': 0.0,
                    'maximum': 0.0
                }
            ]
        }
    }
    db.write(data)

# 添加 LCIA 方法
method = ('Tiangong', 'Environmental_Footprint', 'Acidification')
if method not in methods:
    m = Method(method)
    m.register()
    m.metadata = {
        'unit': 'mol H+ eq',
        'description': 'Acidification potential',
        'abbreviation': 'AP'
    }
    m.write([
        (('biosphere3', 'c1b91234-6f24-417b-8309-46111d09c457'), 1.0)  # NOx = 1.0 mol H+ eq
    ])