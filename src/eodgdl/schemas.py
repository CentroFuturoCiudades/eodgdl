import pandas as pd
import pandera.pandas as pa

mun_col = pa.Column(
    pd.CategoricalDtype(
        [
            "Guadalajara",
            "Tlaquepaque",
            "Tonalá",
            "El Salto",
            "Zapopan",
            "Tlajomulco",
            "Juanacatlán",
            "Zapotlanejo",
            "Ixtlahuacán de los Membrillos",
        ],
        ordered=False,
    )
)

cent_col = pa.Column(
    pd.CategoricalDtype(
        [
            "01",
            "02F",
            "03",
            "04",
            "05",
            "06A",
            "06B",
            "07",
            "08",
            "09",
            "10",
            "11F",
            "12F",
            "13",
            "14F",
            "15",
            "16",
            "17",
            "18",
            "20",
            "21A",
            "21B",
            "22",
            "23",
            "24A",
            "24B",
            "25",
            "26",
            "29F",
            "30",
            "30F",
            "32",
            "34",
            "35",
            "37F",
            "38",
            "39F",
            "40",
            "41",
            "42",
            "43",
            "44",
            "45",
            "46F",
            "48F",
            "49F",
            "50",
            "51F",
            "52",
            "53",
            "54",
            "57",
            "59",
            "60",
            "61",
            "63F",
            "65A",
            "65B",
            "68A",
            "68B",
            "69A",
            "69B",
            "70A",
            "70B",
            "Acceso_Chapala",
            "Acceso_Colotlán",
            "Acceso_López_Mateos",
            "Acceso_Saltillo",
            "Acceso_Vallarta",
            "Acceso_Zapotlanejo",
            "Aeropuerto",
        ],
        ordered=True,
    )
)

si_no_col = pa.Column(pd.CategoricalDtype(["Sí", "No"]))
si_no_null_col = pa.Column(
    pd.CategoricalDtype(["Sí", "No"]),
    checks=pa.Check(lambda s: set(s.dropna().unique()) == set(["Sí", "No"])),
    nullable=True,
)
si_no_null_col_reg = pa.Column(
    pd.CategoricalDtype(["Sí", "No"]),
    checks=pa.Check(lambda s: set(s.dropna().unique()) == set(["Sí", "No"])),
    nullable=True,
    regex=True,
)
ageb_col = pa.Column(str, pa.Check(lambda s: s.str.len().isin([9, 13])))

mode_col = pa.Column(
    pd.CategoricalDtype(
        [
            "AUTOMÓVIL PARTICULAR",
            "TREN LIGERO",
            "A PIE",
            "CAMIÓN O AUTOBÚS",
            "Línea 2 Sitren",
            "Taxi plataforma (Uber/Didi)",
            "MI MACRO",
            "Bicicleta",
            "MOTOCICLETA",
            "Taxi",
            "Transporte de personal",
            "Transporte escolar",
            "Transporte escolar particular",
            "Mi Bici",
            "Servicio suburbano",
            "Taxi colectivo",
            "Transporte informal",
            "Línea 1 Sitren",
            "Línea 3 Sitren",
            "Mototaxi",
        ]
    ),
)

mode_col_nul = pa.Column(
    pd.CategoricalDtype(
        [
            "AUTOMÓVIL PARTICULAR",
            "TREN LIGERO",
            "A PIE",
            "CAMIÓN O AUTOBÚS",
            "Línea 2 Sitren",
            "Taxi plataforma (Uber/Didi)",
            "MI MACRO",
            "Bicicleta",
            "MOTOCICLETA",
            "Taxi",
            "Transporte de personal",
            "Transporte escolar",
            "Transporte escolar particular",
            "Mi Bici",
            "Servicio suburbano",
            "Taxi colectivo",
            "Transporte informal",
            "Línea 1 Sitren",
            "Línea 3 Sitren",
            "Mototaxi",
        ]
    ),
    nullable=True,
)


viv_schema = pa.DataFrameSchema(
    {
        "ponderador": pa.Column(int, pa.Check.greater_than(0)),
        "fecha": pa.Column(pd.DatetimeTZDtype(tz="utc")),
        "municipio": mun_col,
        "ageb": ageb_col,
        "centralidad": cent_col,
        "problema_movilidad_principal": pa.Column(
            pd.CategoricalDtype(
                [
                    "Congestionamiento",
                    "Pavimento deteriorado",
                    "Vialidades insuficientes",
                    "Falta de señalamientos",
                    "Falta de cruces peatonales",
                    "Insuficiente transporte público",
                    "Falta de ciclovías",
                    "Falta de estacionamiento",
                    "Tiempos excesivos de traslado",
                    "Calles inundadas",
                ]
            ),
            nullable=True,
        ),
        "problema_tp_principal": pa.Column(
            pd.CategoricalDtype(
                [
                    "El sobrecupo de personas en las unidades del transporte",
                    "La distancia a las paradas",
                    "Poca frecuencia de paso en la noche",
                    "Trato del/de la chofer",
                    "Falta de cubiertas para espera en las paradas",
                    "El costo del viaje",
                    "La falta de comodidad de las unidades del transporte",
                    "El tiempo de duración de viaje",
                    "Poca frecuencia de paso en el día",
                    "Falta de información de rutas",
                    "Conducción del/ de la chofer",
                ]
            ),
            nullable=True,
        ),
        "personas_en_vivienda": pa.Column(
            pd.CategoricalDtype(
                ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10 y +"], ordered=True
            )
        ),
        "tenencia_vivienda": pa.Column(
            pd.CategoricalDtype(
                ["Rentada", "Prestada", "Propia", "Hipotecada", "Otros"]
            )
        ),
        "ingreso_mensual_hogar": pa.Column(
            pd.CategoricalDtype(
                [
                    "No sabe",
                    "Sin ingresos",
                    "Hasta $1,500",
                    "$1,501 y $3,000",
                    "$3,001 y $7,000",
                    "$7,001 y $10,000",
                    "$10,001 y $15,000",
                    "$15,001 y $25,000",
                    "Más de $25,001",
                    "No quiso responder",
                ],
                ordered=True,
            )
        ),
        "n_autos_camionetas": pa.Column(
            pd.CategoricalDtype(["0", "1", "2", "3", "4 o más"], ordered=True)
        ),
        "n_motos": pa.Column(
            pd.CategoricalDtype(["0", "1", "2", "3", "4 o más"], ordered=True)
        ),
        "n_bicicletas": pa.Column(
            pd.CategoricalDtype(["0", "1", "2", "3", "4 o más"], ordered=True)
        ),
        "donde_estacionan": pa.Column(
            pd.CategoricalDtype(
                [
                    "En la calle, afuera de la casa",
                    "En la cochera de la casa",
                    "En la calle, pero no frente a mi casa",
                    "Estacionamiento privado",
                    "Otro lugar, ¿dónde?",
                    "Estacionamiento público",
                    "Pensión",
                ]
            ),
            nullable=True,
        ),
        "informante_principal": pa.Column(
            pd.CategoricalDtype(
                [
                    "Jefe del hogar",
                    "Otro parentesco",
                    "Hijo",
                    "Cónyuge",
                    "Nieto",
                    "Compañero",
                    "Sin parentesco",
                ]
            ),
            nullable=True,
        ),
        "tiene_internet": si_no_col,
        "dispuesto_participar_futuro": si_no_col,
    },
    strict=False,
    coerce=True,
    index=pa.Index(int, name="folio_vivienda", unique=True, coerce=True),
)

hab_schema = pa.DataFrameSchema(
    {
        "ponderador": pa.Column(int, pa.Check.greater_than(0)),
        "fecha": pa.Column(pd.DatetimeTZDtype(tz="utc")),
        "municipio": mun_col,
        "ageb": ageb_col,
        "centralidad": cent_col,
        "salio_casa_ayer": si_no_col,
        "razon_no_viaje": pa.Column(
            pd.CategoricalDtype(
                [
                    "Otros (especifique)",
                    "Descanso o periodo vacacional",
                    "Maternidad",
                    "Actividades pendientes en casa",
                    "Trabaja desde casa",
                    "Por problemas permanentes de salud",
                    "Por discapacidad",
                    "Por problemas temporales de salud",
                    "Para cuidar a un enfermo",
                    "Adulto Mayor",
                    "Desempleado",
                    "Cuida persona",
                ]
            ),
            nullable=True,
        ),
        "viajes_contados": pa.Column(int),
        "dia_semana_viajes": pa.Column(
            pd.CategoricalDtype(
                ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"], ordered=True
            ),
            nullable=True,
        ),
        "parentesco": pa.Column(
            pd.CategoricalDtype(
                [
                    "Jefe del hogar",
                    "Cónyuge",
                    "Hijo",
                    "Otro parentesco",
                    "Compañero",
                    "Nieto",
                    "Sin parentesco",
                ]
            ),
            nullable=True,
        ),
        "sexo_nacimiento": pa.Column(pd.CategoricalDtype(["Hombres", "Mujeres"])),
        "genero_identidad": pa.Column(
            pd.CategoricalDtype(["Hombres", "Mujeres", "No binario"])
        ),
        "grupo_edad": pa.Column(
            pd.CategoricalDtype(
                ["6-11", "12-17", "18-34", "35-54", "55 y +"],
                ordered=True,
            )
        ),
        "edad": pa.Column(int),
        "estado_civil": pa.Column(
            pd.CategoricalDtype(
                [
                    "Casado",
                    "Soltero",
                    "Unión libre",
                    "Viudo",
                    "Separado",
                    "Divorciado",
                    "Otros (especifique)",
                ]
            ),
            nullable=True,
        ),
        "escolaridad": pa.Column(
            pd.CategoricalDtype(
                [
                    "Maestría o doctorado",
                    "Licenciatura o profesional",
                    "Preparatoria o bachillerato",
                    "Primaria",
                    "Carrera técnica con preparatoria terminada",
                    "Secundaria",
                    "Carrera técnica con secundaria terminada",
                    "Ninguno",
                    "Kinder",
                    "Preescolar",
                    "No sabe",
                    "Normal básica",
                ]
            ),
            nullable=True,
        ),
        "trabajo_semana_pasada": pa.Column(
            pd.CategoricalDtype(
                [
                    "Tiempo completo",
                    "Se dedica a los quehaceres del hogar o a cuidar a sus hijos",
                    "Es estudiante",
                    "No trabajo",
                    "Busco trabajo",
                    "Medio tiempo",
                    "Es jubilado o pensionado",
                    "Está incapacitado permanentemente para trabajar",
                    "Tenía trabajo, pero no trabajó",
                ]
            ),
            nullable=True,
        ),
        "ocupacion": pa.Column(
            pd.CategoricalDtype(
                [
                    "Patrón o empresario",
                    "Hogar",
                    "Estudiante",
                    "Empleado",
                    "Trabajador independiente",
                    "Desempleado",
                    "Jubilado o pensionado",
                    "Trabajador del campo",
                    "Persona trabajadora del hogar",
                    "Profesor",
                ]
            ),
            nullable=True,
        ),
        "giro_empresa": pa.Column(
            pd.CategoricalDtype(
                [
                    "Servicio",
                    "Comercio",
                    "Industria",
                    "Gobierno/sector público",
                    "Educación",
                ]
            ),
            nullable=True,
        ),
        "tiene_discapacidad": si_no_null_col,
        "tipo_discapacidad": pa.Column(
            pd.CategoricalDtype(
                [
                    "Visual",
                    "Física o motora",
                    "Auditiva",
                    "Psicosocial",
                    "Intelectual",
                    "Múltiple",
                    "Sensorial",
                ]
            ),
            nullable=True,
        ),
        "requiere_acompanamiento": si_no_null_col,
        "razon_acompanamiento": pa.Column(
            pd.CategoricalDtype(
                [
                    "Discapacidad permanente",
                    "Problemas de vista",
                    "Problemas de movilidad/discapacidad temporal",
                    "Edad avanzada",
                    "Es un infante",
                    "Down/Autismo",
                    "No escucha",
                    "Otros (especifique)",
                ]
            ),
            nullable=True,
        ),
        "quien_apoya_viajes": pa.Column(
            pd.CategoricalDtype(
                [
                    "Otro familiar",
                    "Papá",
                    "Hijo",
                    "Hermano(a)",
                    "Cónyuge",
                    "Hija",
                    "Mamá",
                    "Pareja",
                    "Amiga",
                    "Enfermera/Pensión",
                    "Otros (especifique)",
                    "Amigo",
                    "Abuelo(a)",
                ]
            ),
            nullable=True,
        ),
        "weekend_.+": si_no_null_col_reg,
    },
    strict=True,
    coerce=True,
    index=pa.MultiIndex(
        [
            pa.Index(int, name="folio_vivienda", unique=True, coerce=True),
            pa.Index(int, name="folio_habitante", coerce=True),
        ],
        unique=True,
        ordered=True,
        coerce=True,
        strict=True,
    ),
)

trips_schema = pa.DataFrameSchema(
    {
        "ponderador": pa.Column(int, pa.Check.greater_than(0)),
        "fecha": pa.Column(pd.DatetimeTZDtype(tz="utc")),
        "municipio": mun_col,
        "ageb": ageb_col,
        "centralidad": cent_col,
        "dia_semana_viajes": pa.Column(
            pd.CategoricalDtype(
                ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"], ordered=True
            ),
        ),
        "tipo_lugar_origen": pa.Column(
            pd.CategoricalDtype(
                [
                    "Su casa",
                    "Comercio, mercado, tienda o centro comercial",
                    "Fábrica o taller",
                    "Escuela",
                    "Oficina",
                    "Deportivo, gimnasio",
                    "Hospital, clínica, consultorio, laboratorio clínico",
                    "Otros (especifique)",
                    "Otra vivienda",
                    "Centro cultural o área recreativa",
                    "Restaurante, bar, cafetería",
                ]
            )
        ),
        "origen": ageb_col,
        "zona_origen": cent_col,
        "hora_inicio_h": pa.Column(pd.Int64Dtype, nullable=True),
        "hora_inicio_m": pa.Column(pd.Int64Dtype, nullable=True),
        "motivo_viaje": pa.Column(
            pd.CategoricalDtype(
                [
                    "Llevar o recoger a alguien",
                    "Regresar a Casa",
                    "Trabajar",
                    "Compras (bienes, productos y servicios)",
                    "Compras (comida)",
                    "Al médico o atención de salud",
                    "Estudiar",
                    "Visitar a alguien",
                    "Deportes o recreación",
                    "Pagar algún servicio o al banco",
                    "Otros (especifique)",
                    "Hacer un trámite",
                    "Guardería",
                ]
            ),
            nullable=True,
        ),
        "tipo_lugar_destino": pa.Column(
            pd.CategoricalDtype(
                [
                    "Escuela",
                    "Su casa",
                    "Oficina",
                    "Comercio, mercado, tienda o centro comercial",
                    "Hospital, clínica, consultorio, laboratorio clínico",
                    "Otra vivienda",
                    "Centro cultural o área recreativa",
                    "Fábrica o taller",
                    "Restaurante, bar, cafetería",
                    "Otros (especifique)",
                    "Deportivo, gimnasio",
                ]
            ),
            nullable=True,
        ),
        "destino": ageb_col,
        "zona_destino": cent_col,
        "n_traslados": pa.Column(int),
        "modo_principal": mode_col,
        "traslado1_medio": mode_col,
        "traslado1_min": pa.Column(int),
        "traslado1_pago": pa.Column(float, nullable=True),
        "traslado2_medio": mode_col_nul,
        "traslado2_min": pa.Column(pd.Int64Dtype, nullable=True),
        "traslado2_pago": pa.Column(float, nullable=True),
        "traslado3_medio": mode_col_nul,
        "traslado3_min": pa.Column(pd.Int64Dtype, nullable=True),
        "traslado3_pago": pa.Column(float, nullable=True),
        "traslado4_medio": mode_col_nul,
        "traslado4_min": pa.Column(pd.Int64Dtype, nullable=True),
        "traslado4_pago": pa.Column(float, nullable=True),
        "traslado5_medio": mode_col_nul,
        "traslado5_min": pa.Column(pd.Int64Dtype, nullable=True),
        "traslado5_pago": pa.Column(float, nullable=True),
        "duracion_total_min": pa.Column(
            pd.CategoricalDtype(
                [
                    "Menos de 6",
                    "6 a 10",
                    "11 a 15",
                    "16 a 30",
                    "31 a 45",
                    "46 a 60",
                    "61 a 90",
                    "91 a 120",
                    "121 y +",
                ],
                ordered=True,
            )
        ),
        "viajo_acompanado": pa.Column(
            pd.CategoricalDtype(
                [
                    "Acompañado algunos traslados",
                    "Acompañado todos los traslados",
                    "Sólo",
                ]
            ),
            nullable=True,
        ),
        "con_quien_viajo": pa.Column(
            pd.CategoricalDtype(
                [
                    "Bebés, niños(as) menores de edad",
                    "Otra persona",
                    "Personas que requieren apoyo para sus traslados",
                ]
            ),
            nullable=True,
        ),
        "rol_en_vehiculo": pa.Column(
            pd.CategoricalDtype(["Conductor", "Acompañante"]), nullable=True
        ),
        "ocupantes_auto_moto": pa.Column(
            pd.CategoricalDtype(["1", "2", "3", "4", "5", "6 o más"], ordered=True),
            nullable=True,
        ),
        "acompanante_fuera_hogar": si_no_null_col,
        "estacionamiento_lugar": pa.Column(
            pd.CategoricalDtype(
                [
                    "No me estacioné",
                    "Cochera propia",
                    "Vía pública",
                    "Estacionamiento privado",
                    "Estacionamiento público",
                    "No sé",
                ]
            ),
            nullable=True,
        ),
        "pago_estacionamiento": pa.Column(
            pd.CategoricalDtype(["No", "Sí", "No sé"]), nullable=True
        ),
        "estacionamiento_pago_total": pa.Column(pd.Int64Dtype, nullable=True),
        "estacionamiento_pago_tipo": pa.Column(
            pd.CategoricalDtype(["Informal", "Formal"]), nullable=True
        ),
        "estacionamiento_frecuencia": pa.Column(
            pd.CategoricalDtype(
                [
                    "Por hora",
                    "Por día",
                    "Por mes",
                    "Otra periocidad, ¿cuál?",
                    "Por semana",
                ]
            ),
            nullable=True,
        ),
        "parentesco": pa.Column(
            pd.CategoricalDtype(
                [
                    "Jefe del hogar",
                    "Cónyuge",
                    "Hijo",
                    "Otro parentesco",
                    "Nieto",
                    "Compañero",
                    "Sin parentesco",
                ]
            ),
            nullable=True,
        ),
        "sexo_nacimiento": pa.Column(pd.CategoricalDtype(["Hombres", "Mujeres"])),
        "genero_identidad": pa.Column(
            pd.CategoricalDtype(["Hombres", "Mujeres", "No binario"])
        ),
        "grupo_edad": pa.Column(
            pd.CategoricalDtype(
                ["6-11", "12-17", "18-34", "35-54", "55 y +"],
                ordered=True,
            )
        ),
        "edad": pa.Column(int),
        "estado_civil": pa.Column(
            pd.CategoricalDtype(
                [
                    "Casado",
                    "Soltero",
                    "Unión libre",
                    "Viudo",
                    "Separado",
                    "Divorciado",
                    "Otros (especifique)",
                ]
            ),
            nullable=True,
        ),
        "escolaridad": pa.Column(
            pd.CategoricalDtype(
                [
                    "Maestría o doctorado",
                    "Licenciatura o profesional",
                    "Preparatoria o bachillerato",
                    "Primaria",
                    "Carrera técnica con preparatoria terminada",
                    "Secundaria",
                    "Carrera técnica con secundaria terminada",
                    "Ninguno",
                    "Kinder",
                    "Preescolar",
                    "No sabe",
                    "Normal básica",
                ]
            ),
            nullable=True,
        ),
        "trabajo_semana_pasada": pa.Column(
            pd.CategoricalDtype(
                [
                    "Tiempo completo",
                    "Se dedica a los quehaceres del hogar o a cuidar a sus hijos",
                    "Es estudiante",
                    "No trabajo",
                    "Busco trabajo",
                    "Medio tiempo",
                    "Es jubilado o pensionado",
                    "Está incapacitado permanentemente para trabajar",
                    "Tenía trabajo, pero no trabajó",
                ]
            ),
            nullable=True,
        ),
        "ocupacion": pa.Column(
            pd.CategoricalDtype(
                [
                    "Patrón o empresario",
                    "Hogar",
                    "Estudiante",
                    "Empleado",
                    "Trabajador independiente",
                    "Desempleado",
                    "Jubilado o pensionado",
                    "Trabajador del campo",
                    "Persona trabajadora del hogar",
                    "Profesor",
                ]
            ),
            nullable=True,
        ),
        "giro_empresa": pa.Column(
            pd.CategoricalDtype(
                [
                    "Servicio",
                    "Comercio",
                    "Industria",
                    "Gobierno/sector público",
                    "Educación",
                ]
            ),
            nullable=True,
        ),
        "tiene_discapacidad": si_no_null_col,
        "tipo_discapacidad": pa.Column(
            pd.CategoricalDtype(
                [
                    "Visual",
                    "Física o motora",
                    "Auditiva",
                    "Psicosocial",
                    "Intelectual",
                    "Múltiple",
                    "Sensorial",
                ]
            ),
            nullable=True,
        ),
    },
    strict=True,
    coerce=True,
    index=pa.MultiIndex(
        [
            pa.Index(int, name="folio_vivienda", unique=True, coerce=True),
            pa.Index(int, name="folio_habitante", coerce=True),
            pa.Index(int, name="folio_viaje", coerce=True),
        ],
        unique=True,
        ordered=True,
        coerce=True,
        strict=True,
    ),
)
