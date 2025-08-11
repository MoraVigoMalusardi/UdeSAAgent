import os
import sys
import pandas as pd
import json
from typing import List, Dict, Optional
from datetime import datetime, time
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

from src.lg_nodes import combi_schedule_tool

with open(os.path.join(project_root, "data/toolCombis/combis.json"), encoding="utf-8") as f:
    GROUND_TRUTH_SCHEDULES = json.load(f)

def extract_time_from_response(response: str) -> Optional[time]:
    """Extrae la hora de una respuesta que menciona 'sale a las HH:MM'"""
    match = re.search(r'sale a las (\d{1,2}:\d{2})', response)
    if match:
        try:
            return datetime.strptime(match.group(1), "%H:%M").time()
        except ValueError:
            return None
    return None

def extract_all_times_from_response(response: str) -> List[time]:
    """Extrae todas las horas mencionadas en formato HH:MM"""
    times = []
    matches = re.findall(r'\b(\d{1,2}:\d{2})\b', response)
    for match in matches:
        try:
            times.append(datetime.strptime(match, "%H:%M").time())
        except ValueError:
            continue
    return times

def detect_station_in_response(response: str) -> Optional[str]:
    """Detecta qué estación se menciona en la respuesta"""
    response_lower = response.lower()
    if "campus udesa" in response_lower:
        return "Campus UdeSA"
    elif "estación victoria" in response_lower or "estacion victoria" in response_lower:
        return "Estación Victoria"
    return None

def is_location_error(response: str) -> bool:
    response_lower = response.lower()
    return any(fragment in response_lower for fragment in [
        "no entendí", "por favor menciona", "no se detectó", "no encontré la parada"
    ])

class CombiToolIntrinsicEvaluator:
    def __init__(self):
        self.results = []
        
    def test_station_recognition(self, test_cases: List[Dict]) -> List[Dict]:
        """Evalúa si la tool reconoce correctamente las estaciones"""
        results = []
        
        for case in test_cases:
            query = case["query"]
            expected_station = case["expected_station"]
            
            try:
                response = combi_schedule_tool.invoke({
                    "query": query,
                    "current_time_str": "14:00"  # Hora fija para consistencia
                })
                
                response_str = str(response)

                if is_location_error(response_str):
                    detected_station = None
                else:
                    detected_station = detect_station_in_response(response_str)

                station_correct = detected_station == expected_station
                
                results.append({
                    "test_type": "station_recognition",
                    "query": query,
                    "expected_station": expected_station,
                    "detected_station": detected_station,
                    "response": str(response),
                    "correct": station_correct
                })
                
            except Exception as e:
                results.append({
                    "test_type": "station_recognition",
                    "query": query,
                    "expected_station": expected_station,
                    "detected_station": None,
                    "response": f"ERROR: {str(e)}",
                    "correct": False
                })
        
        return results
    
    def test_schedule_completeness(self, test_cases: List[Dict]) -> List[Dict]:
        """Evalúa si devuelve todos los horarios correctos para cada estación"""
        results = []
        
        for case in test_cases:
            query = case["query"]
            expected_station = case["expected_station"]
            
            try:
                response = combi_schedule_tool.invoke({
                    "query": query,
                    "current_time_str": "08:00"  #Hora temprana para ver todos los horarios
                })
                
                response_str = str(response)
                extracted_times = extract_all_times_from_response(response_str)
                
                if expected_station in GROUND_TRUTH_SCHEDULES:
                    expected_times = []
                    for time_str in GROUND_TRUTH_SCHEDULES[expected_station]:
                        try:
                            expected_times.append(datetime.strptime(time_str, "%H:%M").time())
                        except ValueError:
                            continue
                    
                    missing_times = [t for t in expected_times if t not in extracted_times]
                    extra_times = [t for t in extracted_times if t not in expected_times]
                    
                    completeness_score = len(set(expected_times) - set(missing_times)) / len(expected_times) if expected_times else 0
                    results.append({
                        "test_type": "schedule_completeness",
                        "query": query,
                        "station": expected_station,
                        "expected_count": len(expected_times),
                        "extracted_count": len(extracted_times),
                        "missing_times": [t.strftime("%H:%M") for t in missing_times],
                        "extra_times": [t.strftime("%H:%M") for t in extra_times],
                        "completeness_score": completeness_score,
                        "response": response_str,
                        "correct": len(missing_times) == 0 and len(extra_times) == 0
                    })
                else:
                    results.append({
                        "test_type": "schedule_completeness",
                        "query": query,
                        "station": expected_station,
                        "response": response_str,
                        "correct": False,
                        "error": f"Station {expected_station} not found in ground truth"
                    })
                    
            except Exception as e:
                results.append({
                    "test_type": "schedule_completeness",
                    "query": query,
                    "station": expected_station,
                    "response": f"ERROR: {str(e)}",
                    "correct": False
                })
        
        return results
    
    def test_next_combi_logic(self, test_cases: List[Dict]) -> List[Dict]:
        """Evalúa la lógica de 'próxima combi'"""
        results = []
        
        for case in test_cases:
            query = case["query"]
            current_time_str = case["current_time_str"]
            expected_station = case["expected_station"]
            expected_next_time = case.get("expected_next_time")  
            
            try:
                response = combi_schedule_tool.invoke({
                    "query": query,
                    "current_time_str": current_time_str
                })
                
                response_str = str(response)
                
                if expected_next_time is None:
                    no_more_combis = "No hay más combis programadas" in response_str
                    results.append({
                        "test_type": "next_combi_logic",
                        "query": query,
                        "current_time": current_time_str,
                        "station": expected_station,
                        "expected_next": "No más combis",
                        "response": response_str,
                        "correct": no_more_combis
                    })
                else:
                    extracted_time = extract_time_from_response(response_str)
                    expected_time_obj = datetime.strptime(expected_next_time, "%H:%M").time()
                    
                    time_correct = extracted_time == expected_time_obj
                    
                    results.append({
                        "test_type": "next_combi_logic",
                        "query": query,
                        "current_time": current_time_str,
                        "station": expected_station,
                        "expected_next": expected_next_time,
                        "extracted_next": extracted_time.strftime("%H:%M") if extracted_time else None,
                        "response": response_str,
                        "correct": time_correct
                    })
                    
            except Exception as e:
                results.append({
                    "test_type": "next_combi_logic",
                    "query": query,
                    "current_time": current_time_str,
                    "station": expected_station,
                    "response": f"ERROR: {str(e)}",
                    "correct": False
                })
        
        return results
    
    def run_full_evaluation(self) -> pd.DataFrame:
        """Ejecuta todos los tests de evaluación intrínseca"""
        
        print("Evaluando reconocimiento de estaciones...")
        station_tests = [
            {"query": "horarios desde Campus UdeSA", "expected_station": "Campus UdeSA"},
            {"query": "combis del campus udesa", "expected_station": "Campus UdeSA"},
            {"query": "desde Estación Victoria", "expected_station": "Estación Victoria"},
            {"query": "estacion victoria horarios", "expected_station": "Estación Victoria"},
            {"query": "horarios de combi", "expected_station": None},  
            {"query": "desde el campus", "expected_station": "Campus UdeSA"},  
        ]
        
        station_results = self.test_station_recognition(station_tests)
        
        print("Evaluando completitud de horarios...")
        schedule_tests = [
            {"query": "todos los horarios desde Campus UdeSA", "expected_station": "Campus UdeSA"},
            {"query": "horarios completos Estación Victoria", "expected_station": "Estación Victoria"},
        ]
        
        schedule_results = self.test_schedule_completeness(schedule_tests)
        
        print("Evaluando lógica de próxima combi...")
        next_combi_tests = [
            # Campus UdeSA tests
            {"query": "próxima combi Campus UdeSA", "current_time_str": "10:00", 
             "expected_station": "Campus UdeSA", "expected_next_time": "10:45"},
            {"query": "próxima combi Campus UdeSA", "current_time_str": "12:30", 
             "expected_station": "Campus UdeSA", "expected_next_time": "12:40"},
            {"query": "próxima combi Campus UdeSA", "current_time_str": "21:00", 
             "expected_station": "Campus UdeSA", "expected_next_time": None},  # No más combis
            
            # Estación Victoria tests  
            {"query": "próxima combi Estación Victoria", "current_time_str": "09:00", 
             "expected_station": "Estación Victoria", "expected_next_time": "09:40"},
            {"query": "próxima combi Estación Victoria", "current_time_str": "13:30", 
             "expected_station": "Estación Victoria", "expected_next_time": "13:50"},
            {"query": "próxima combi Estación Victoria", "current_time_str": "19:00", 
             "expected_station": "Estación Victoria", "expected_next_time": None},  # No más combis
        ]
        
        next_combi_results = self.test_next_combi_logic(next_combi_tests)
        
        all_results = station_results + schedule_results + next_combi_results
        df = pd.DataFrame(all_results)
        
        return df
    
    def generate_report(self, df: pd.DataFrame, output_path: str):
        """Genera reporte detallado de la evaluación"""
        
        df.to_csv(output_path, index=False)
        
        print(f"\nREPORTE DE EVALUACIÓN INTRÍNSECA - COMBI TOOL")
        print(f"=" * 60)
        
        total_tests = len(df)
        passed_tests = df['correct'].sum()
        accuracy = passed_tests / total_tests
        
        print(f"Total de tests: {total_tests}")
        print(f"Tests pasados: {passed_tests}")
        print(f"Precisión general: {accuracy:.2%}")
        
        print(f"\nPRECISIÓN POR TIPO DE TEST:")
        for test_type in df['test_type'].unique():
            subset = df[df['test_type'] == test_type]
            type_accuracy = subset['correct'].mean()
            print(f"  {test_type}: {type_accuracy:.2%} ({subset['correct'].sum()}/{len(subset)})")
        
        failed_tests = df[df['correct'] == False]
        if not failed_tests.empty:
            print(f"\nTESTS FALLIDOS ({len(failed_tests)}):")
            for _, row in failed_tests.iterrows():
                print(f"  • {row['test_type']}: {row['query']}")
                if 'expected_station' in row and pd.notna(row['expected_station']):
                    print(f"    Esperado: {row['expected_station']}")
                if 'detected_station' in row and pd.notna(row['detected_station']):
                    print(f"    Obtenido: {row['detected_station']}")
                print()
        
        print(f"\nResultados guardados en: {output_path}")

if __name__ == "__main__":
    output_file = os.path.join(project_root, "evaluations", "combi_tool", "results", f"combi_tool_test.csv")

    
    evaluator = CombiToolIntrinsicEvaluator()
    results_df = evaluator.run_full_evaluation()
    evaluator.generate_report(results_df, output_file)
