import os
import sys
import pandas as pd
import json
from typing import List, Dict
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

from src.lg_nodes import exam_schedule_and_modality_tool

with open(os.path.join(project_root, "data/toolExamenesParciales/parcialesOtoño2025.json"), encoding="utf-8") as f:
    GROUND_TRUTH_EXAMS = json.load(f)

def extract_exam_info_from_response(response: str) -> Dict:
    """Extrae información del examen de la respuesta"""
    info = {
        "fecha": None,
        "hora": None,
        "aula": None,
        "modalidad": None,
        "codigo_materia": None,
        "nombre_materia": None
    }
    

    fecha_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', response)
    if fecha_match:
        info["fecha"] = fecha_match.group(1)
    
    hora_match = re.search(r'Horario:\s*([^\\n]+)', response)
    if hora_match:
        info["hora"] = hora_match.group(1).strip()
    
    aula_match = re.search(r'Aula:\s*([^\\n]+)', response)
    if aula_match:
        info["aula"] = aula_match.group(1).strip()
    
    modalidad_match = re.search(r'Modalidad:\s*([^\\n]+)', response)
    if modalidad_match:
        info["modalidad"] = modalidad_match.group(1).strip()
    
    codigo_match = re.search(r'([A-Z]+\d+)', response)
    if codigo_match:
        info["codigo_materia"] = codigo_match.group(1)
    
    return info

def detect_subject_recognition(response: str, expected_subject: str) -> bool:
    """Verifica si la materia esperada aparece en la respuesta"""
    response_lower = response.lower()
    expected_lower = expected_subject.lower()
    
    return expected_lower in response_lower or any(word in response_lower for word in expected_lower.split())

def is_exam_error(response: str) -> bool:
    """Detecta si la respuesta indica que no se encontró información"""
    response_lower = response.lower()
    return any(fragment in response_lower for fragment in [
        "no entendí", "perdón", "no encontré", "no hay fechas", "no se encontraron"
    ])

class ExamToolIntrinsicEvaluator:
    def __init__(self):
        self.results = []
        
    def test_subject_recognition(self, test_cases: List[Dict]) -> List[Dict]:
        """Evalúa si la tool reconoce correctamente las materias por código y nombre"""
        results = []
        
        for case in test_cases:
            query = case["query"]
            expected_subject = case["expected_subject"]
            expected_code = case.get("expected_code")
            current_date = case.get("current_date_str", "01/03/2025")
            
            try:
                response = exam_schedule_and_modality_tool.invoke({
                    "query": query,
                    "current_date_str": current_date
                })
                
                response_str = str(response)
                
                if is_exam_error(response_str):
                    subject_recognized = False
                else:
                    subject_recognized = detect_subject_recognition(response_str, expected_subject)
                    if expected_code and not subject_recognized:
                        subject_recognized = expected_code.upper() in response_str.upper()
                
                results.append({
                    "test_type": "subject_recognition",
                    "query": query,
                    "expected_subject": expected_subject,
                    "expected_code": expected_code,
                    "subject_recognized": subject_recognized,
                    "response": response_str,
                    "correct": subject_recognized
                })
                
            except Exception as e:
                results.append({
                    "test_type": "subject_recognition",
                    "query": query,
                    "expected_subject": expected_subject,
                    "response": f"ERROR: {str(e)}",
                    "correct": False
                })
        
        return results
    
    def test_modality_detection(self, test_cases: List[Dict]) -> List[Dict]:
        """Evalúa si la tool detecta correctamente las modalidades de examen"""
        results = []
        
        for case in test_cases:
            query = case["query"]
            expected_modality = case["expected_modality"]  # "presencial", "domiciliario", "sin examen"
            current_date = case.get("current_date_str", "01/03/2025")
            
            try:
                response = exam_schedule_and_modality_tool.invoke({
                    "query": query,
                    "current_date_str": current_date
                })
                
                response_str = str(response)
                
                #detectar modalidad
                if "domiciliario" in response_str.lower():
                    detected_modality = "domiciliario"
                elif "sin examen" in response_str.lower():
                    detected_modality = "sin examen"
                elif "presencial" in response_str.lower() or ("fecha:" in response_str.lower() and "aula:" in response_str.lower()):
                    detected_modality = "presencial"
                else:
                    detected_modality = None
                
                modality_correct = detected_modality == expected_modality
                
                results.append({
                    "test_type": "modality_detection",
                    "query": query,
                    "expected_modality": expected_modality,
                    "detected_modality": detected_modality,
                    "response": response_str,
                    "correct": modality_correct
                })
                
            except Exception as e:
                results.append({
                    "test_type": "modality_detection",
                    "query": query,
                    "expected_modality": expected_modality,
                    "response": f"ERROR: {str(e)}",
                    "correct": False
                })
        
        return results
    
    def test_date_and_venue_accuracy(self, test_cases: List[Dict]) -> List[Dict]:
        """Evalúa la precisión de fechas, horarios y aulas"""
        results = []
        
        for case in test_cases:
            query = case["query"]
            expected_info = case["expected_info"]  
            current_date = case.get("current_date_str", "01/03/2025")
            
            try:
                response = exam_schedule_and_modality_tool.invoke({
                    "query": query,
                    "current_date_str": current_date
                })
                
                response_str = str(response)
                extracted_info = extract_exam_info_from_response(response_str)
                
                fecha_correct = extracted_info["fecha"] == expected_info.get("fecha")
                hora_correct = extracted_info["hora"] == expected_info.get("hora")
                aula_correct = extracted_info["aula"] == expected_info.get("aula")
                
                #error de presicion
                fields_to_check = ["fecha", "hora", "aula"]
                correct_fields = sum([
                    fecha_correct,
                    hora_correct,
                    aula_correct
                ])
                
                accuracy_score = correct_fields / len(fields_to_check)
                
                results.append({
                    "test_type": "date_and_venue_accuracy",
                    "query": query,
                    "expected_info": expected_info,
                    "extracted_info": extracted_info,
                    "fecha_correct": fecha_correct,
                    "hora_correct": hora_correct,
                    "aula_correct": aula_correct,
                    "accuracy_score": accuracy_score,
                    "response": response_str,
                    "correct": accuracy_score >= 0.8  
                })
                
            except Exception as e:
                results.append({
                    "test_type": "date_and_venue_accuracy",
                    "query": query,
                    "expected_info": expected_info,
                    "response": f"ERROR: {str(e)}",
                    "correct": False
                })
        
        return results
    
    def test_year_filtering(self, test_cases: List[Dict]) -> List[Dict]:
        """Evalúa si la tool filtra correctamente por año"""
        results = []
        
        for case in test_cases:
            query = case["query"]
            expected_year = case["expected_year"]
            should_find_results = case["should_find_results"]
            current_date = case.get("current_date_str", "01/03/2025")
            
            try:
                response = exam_schedule_and_modality_tool.invoke({
                    "query": query,
                    "current_date_str": current_date
                })
                
                response_str = str(response)
                
                found_results = not is_exam_error(response_str) and len(response_str.strip()) > 50
                
                year_mentioned = str(expected_year) in response_str if expected_year else True
                
                filtering_correct = (found_results == should_find_results) and year_mentioned
                
                results.append({
                    "test_type": "year_filtering",
                    "query": query,
                    "expected_year": expected_year,
                    "should_find_results": should_find_results,
                    "found_results": found_results,
                    "year_mentioned": year_mentioned,
                    "response": response_str,
                    "correct": filtering_correct
                })
                
            except Exception as e:
                results.append({
                    "test_type": "year_filtering",
                    "query": query,
                    "expected_year": expected_year,
                    "response": f"ERROR: {str(e)}",
                    "correct": False
                })
        
        return results
    
    def run_full_evaluation(self) -> pd.DataFrame:
        """Ejecuta todos los tests de evaluación intrínseca"""
        
        print("Evaluando reconocimiento de materias...")
        subject_tests = [
            {"query": "examen de Fundamentos de Administración", "expected_subject": "Fundamentos de Administración", "expected_code": "A016"},
            {"query": "parcial de Matemática I", "expected_subject": "Matemática I", "expected_code": "C030"},
            {"query": "examen de Física I", "expected_subject": "Física I", "expected_code": "I208"},
            {"query": "final de P042", "expected_subject": "Filosofía Política Clásica", "expected_code": "P042"},
        ]
        
        subject_results = self.test_subject_recognition(subject_tests)
        
        print("Evaluando detección de modalidades...")
        modality_tests = [
            {"query": "modalidad del examen de A016", "expected_modality": "domiciliario"},  # Basado en el JSON
            {"query": "examen de Fundamentos de Contabilidad", "expected_modality": "presencial"},  # Tiene fecha concreta
            {"query": "examen domiciliario de Marketing", "expected_modality": "domiciliario"},
            {"query": "materias sin examen", "expected_modality": "sin examen"},
        ]
        
        modality_results = self.test_modality_detection(modality_tests)
        
        print("Evaluando precisión de fechas y aulas...")
        date_venue_tests = [
            {
                "query": "examen de Fundamentos de Contabilidad",  
                "expected_info": {
                    "fecha": "22/04/2025",
                    "hora": "17hs",
                    "aula": "HAM / H005 / H011"
                }
            }
        ]
        
        date_venue_results = self.test_date_and_venue_accuracy(date_venue_tests)

        

        all_results = subject_results + modality_results + date_venue_results 
        df = pd.DataFrame(all_results)
        
        return df
    
    def generate_report(self, df: pd.DataFrame, output_path: str):
        """Genera reporte detallado de la evaluación"""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        
        print(f"\nREPORTE DE EVALUACIÓN ESPECÍFICA - EXAM TOOL")
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
                if 'expected_subject' in row and pd.notna(row['expected_subject']):
                    print(f"    Esperado: {row['expected_subject']}")
                if 'expected_modality' in row and pd.notna(row['expected_modality']):
                    print(f"    Modalidad esperada: {row['expected_modality']}")
                print()
        
        print(f"\nResultados guardados en: {output_path}")

if __name__ == "__main__":
    output_file = os.path.join(project_root, "evaluations", "exams_tool", "results", f"exams_tool_test.csv")
    
    evaluator = ExamToolIntrinsicEvaluator()
    results_df = evaluator.run_full_evaluation()
    evaluator.generate_report(results_df, output_file)