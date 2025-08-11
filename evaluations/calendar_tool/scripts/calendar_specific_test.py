import os
import sys
import pandas as pd
import json
from typing import List, Dict, Optional
from datetime import datetime, date
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

from src.lg_nodes import academic_calendar_tool

with open(os.path.join(project_root, "data/toolCalendarioAcademico/calendarioAcademico2025.json"), encoding="utf-8") as f:
    GROUND_TRUTH_CALENDAR = json.load(f)

def extract_dates_from_response(response: str) -> List[date]:
    """Extrae fechas en formato DD/MM/YYYY de la respuesta"""
    dates = []
    matches = re.findall(r'\b(\d{1,2}/\d{1,2}/\d{4})\b', response)
    for match in matches:
        try:
            dates.append(datetime.strptime(match, "%d/%m/%Y").date())
        except ValueError:
            continue
    return dates

def extract_event_names_from_response(response: str) -> List[str]:
    """Extrae nombres de eventos de la respuesta (líneas que empiezan con -)"""
    events = []
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('- '):
            event_name = line[2:].split('(')[0].split('—')[0].strip()
            events.append(event_name)
    return events

def detect_event_type_in_response(response: str) -> Optional[str]:
    """Detecta qué tipo de evento se menciona en la respuesta"""
    response_lower = response.lower()
    tipos = ["feriado", "parciales", "finales", "vacaciones", "inicio", "recuperatorios", "plazo"]
    
    for tipo in tipos:
        if tipo in response_lower:
            return tipo
    return None

def is_calendar_error(response: str) -> bool:
    """Detecta si la respuesta indica que no se encontraron eventos"""
    response_lower = response.lower()
    return any(fragment in response_lower for fragment in [
        "no encontré eventos", "no hay eventos", "no se encontró", "no coincidan"
    ])

class CalendarToolIntrinsicEvaluator:
    def __init__(self):
        self.results = []
        
    def test_event_type_recognition(self, test_cases: List[Dict]) -> List[Dict]:
        """Evalúa si la tool reconoce correctamente los tipos de eventos"""
        results = []
        
        for case in test_cases:
            query = case["query"]
            expected_type = case["expected_type"]
            current_date = case.get("current_date_str", "01/03/2025")  # Fecha por defecto
            
            try:
                response = academic_calendar_tool.invoke({
                    "query": query,
                    "current_date_str": current_date
                })
                
                response_str = str(response)
                
                if is_calendar_error(response_str):
                    detected_type = None
                else:
                    detected_type = detect_event_type_in_response(response_str)
                
                type_correct = detected_type == expected_type
                
                results.append({
                    "test_type": "event_type_recognition",
                    "query": query,
                    "expected_type": expected_type,
                    "detected_type": detected_type,
                    "current_date": current_date,
                    "response": response_str,
                    "correct": type_correct
                })
                
            except Exception as e:
                results.append({
                    "test_type": "event_type_recognition",
                    "query": query,
                    "expected_type": expected_type,
                    "detected_type": None,
                    "response": f"ERROR: {str(e)}",
                    "correct": False
                })
        
        return results
    
    def test_semester_filtering(self, test_cases: List[Dict]) -> List[Dict]:
        """Evalúa si la tool filtra correctamente por semestre"""
        results = []
        
        for case in test_cases:
            query = case["query"]
            expected_semester = case["expected_semester"]
            current_date = case.get("current_date_str", "01/03/2025")
            
            try:
                response = academic_calendar_tool.invoke({
                    "query": query,
                    "current_date_str": current_date
                })
                
                response_str = str(response)
                
                #me fijo si la respuesta tiene el semestre correcto, sino que muestre ambos
                if expected_semester == 1:
                    semester_correct = "semestre 1" in response_str.lower()
                elif expected_semester == 2:
                    semester_correct = "semestre 2" in response_str.lower()
                else:
                    semester_correct = "semestre" in response_str.lower()
                
                results.append({
                    "test_type": "semester_filtering",
                    "query": query,
                    "expected_semester": expected_semester,
                    "current_date": current_date,
                    "response": response_str,
                    "correct": semester_correct
                })
                
            except Exception as e:
                results.append({
                    "test_type": "semester_filtering",
                    "query": query,
                    "expected_semester": expected_semester,
                    "response": f"ERROR: {str(e)}",
                    "correct": False
                })
        
        return results
    
    def test_next_event_logic(self, test_cases: List[Dict]) -> List[Dict]:
        """Evalúa la lógica de 'próximo evento'"""
        results = []
        
        for case in test_cases:
            query = case["query"]
            current_date_str = case["current_date_str"]
            expected_has_next = case["expected_has_next"]  
            
            try:
                response = academic_calendar_tool.invoke({
                    "query": query,
                    "current_date_str": current_date_str
                })
                
                response_str = str(response)
                
                has_proximo_text = "próximo evento" in response_str.lower()
                
                if expected_has_next:
                    logic_correct = has_proximo_text and not is_calendar_error(response_str)
                else:
                    logic_correct = not has_proximo_text or is_calendar_error(response_str)
                
                results.append({
                    "test_type": "next_event_logic",
                    "query": query,
                    "current_date": current_date_str,
                    "expected_has_next": expected_has_next,
                    "has_proximo_text": has_proximo_text,
                    "response": response_str,
                    "correct": logic_correct
                })
                
            except Exception as e:
                results.append({
                    "test_type": "next_event_logic",
                    "query": query,
                    "current_date": current_date_str,
                    "expected_has_next": expected_has_next,
                    "response": f"ERROR: {str(e)}",
                    "correct": False
                })
        
        return results
    
    def test_date_accuracy(self, test_cases: List[Dict]) -> List[Dict]:
        """Evalúa la precisión de las fechas devueltas"""
        results = []
        
        for case in test_cases:
            query = case["query"]
            expected_events = case["expected_events"]  
            current_date = case.get("current_date_str", "01/03/2025")
            
            try:
                response = academic_calendar_tool.invoke({
                    "query": query,
                    "current_date_str": current_date
                })
                
                response_str = str(response)
                extracted_events = extract_event_names_from_response(response_str)
                extracted_dates = extract_dates_from_response(response_str)
                
                events_found = []
                events_missing = []
                
                for expected_event in expected_events:
                    found = any(expected_event.lower() in event.lower() for event in extracted_events)
                    if found:
                        events_found.append(expected_event)
                    else:
                        events_missing.append(expected_event)
                
                accuracy_score = len(events_found) / len(expected_events) if expected_events else 1.0
                
                results.append({
                    "test_type": "date_accuracy",
                    "query": query,
                    "expected_events": expected_events,
                    "extracted_events": extracted_events,
                    "events_found": events_found,
                    "events_missing": events_missing,
                    "accuracy_score": accuracy_score,
                    "dates_count": len(extracted_dates),
                    "response": response_str,
                    "correct": len(events_missing) == 0
                })
                
            except Exception as e:
                results.append({
                    "test_type": "date_accuracy",
                    "query": query,
                    "expected_events": expected_events,
                    "response": f"ERROR: {str(e)}",
                    "correct": False
                })
        
        return results
    
    def run_full_evaluation(self) -> pd.DataFrame:
        """Ejecuta todos los tests de evaluación intrínseca"""
        
        print("valuando reconocimiento de tipos de eventos...")
        event_type_tests = [
            {"query": "¿Cuáles son los feriados de 2025?", "expected_type": "feriado"},
            {"query": "fechas de parciales", "expected_type": "parciales"},
            {"query": "cuándo son los finales", "expected_type": "finales"},
            {"query": "período de vacaciones", "expected_type": "vacaciones"},
            {"query": "inicio de clases", "expected_type": "inicio"},
            {"query": "recuperatorios cuando son", "expected_type": "recuperatorios"},
            {"query": "plazos importantes", "expected_type": "plazo"},
        ]
        
        event_type_results = self.test_event_type_recognition(event_type_tests)
        
        print("Evaluando filtrado por semestre...")
        semester_tests = [
            {"query": "eventos del semestre 1", "expected_semester": 1},
            {"query": "fechas importantes semestre 2", "expected_semester": 2},
            {"query": "feriados primer semestre", "expected_semester": 1},
            {"query": "parciales segundo semestre", "expected_semester": 2},
            {"query": "todos los eventos", "expected_semester": None},  # Ambos semestres
        ]
        
        semester_results = self.test_semester_filtering(semester_tests)
        
        print("Evaluando lógica de próximo evento...")
        next_event_tests = [
            {"query": "próximo feriado", "current_date_str": "01/03/2025", "expected_has_next": True},
            {"query": "próximo evento académico", "current_date_str": "01/03/2025", "expected_has_next": True},
        ]
        
        next_event_results = self.test_next_event_logic(next_event_tests)
        
        print("Evaluando precisión de fechas...")
        date_accuracy_tests = [
            {
                "query": "feriados de abril 2025",
                "expected_events": ["Feriado nacional", "Feriado trasladable", "Feriado puente"],  
                "current_date_str": "01/03/2025"
            },
            {
                "query": "fechas de parciales semestre 1",
                "expected_events": ["Parciales"],  
                "current_date_str": "01/03/2025"
            },
        ]
        
        date_accuracy_results = self.test_date_accuracy(date_accuracy_tests)
        
        all_results = event_type_results + semester_results + next_event_results + date_accuracy_results
        df = pd.DataFrame(all_results)
        
        return df
    
    def generate_report(self, df: pd.DataFrame, output_path: str):
        """Genera reporte detallado de la evaluación"""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        
        print(f"\nREPORTE DE EVALUACIÓN ESPECÍFICA - CALENDAR TOOL")
        print(f"=" * 60)
        
        total_tests = len(df)
        passed_tests = df['correct'].sum()
        accuracy = passed_tests / total_tests
        
        print(f"Total de tests: {total_tests}")
        print(f"Tests pasados: {passed_tests}")
        print(f"Precisión general: {accuracy:.2%}")
        
        print(f"\nRECISIÓN POR TIPO DE TEST:")
        for test_type in df['test_type'].unique():
            subset = df[df['test_type'] == test_type]
            type_accuracy = subset['correct'].mean()
            print(f"  {test_type}: {type_accuracy:.2%} ({subset['correct'].sum()}/{len(subset)})")
        
        failed_tests = df[df['correct'] == False]
        if not failed_tests.empty:
            print(f"\nTESTS FALLIDOS ({len(failed_tests)}):")
            for _, row in failed_tests.iterrows():
                print(f"  • {row['test_type']}: {row['query']}")
                if 'expected_type' in row and pd.notna(row['expected_type']):
                    print(f"    Esperado: {row['expected_type']}")
                if 'detected_type' in row and pd.notna(row['detected_type']):
                    print(f"    Obtenido: {row['detected_type']}")
                print()
        
        print(f"\nResultados guardados en: {output_path}")

if __name__ == "__main__":
    output_file = os.path.join(project_root, "evaluations", "calendar_tool", "results", f"calendar_tool_test.csv")

    
    evaluator = CalendarToolIntrinsicEvaluator()
    results_df = evaluator.run_full_evaluation()
    evaluator.generate_report(results_df, output_file)